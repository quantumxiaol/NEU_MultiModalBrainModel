import copy
import os
import random
import time

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import lr_scheduler

from my_models import SimpleTransformerModel

load_dotenv()
device = torch.device(os.getenv("DEVICE", "cpu"))
print(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_connectivity(x, y, normal_label=0):
    finite_mask = np.isfinite(x)
    normal_subject_mask = y == normal_label
    reference_x = x[normal_subject_mask] if np.any(normal_subject_mask) else x
    reference_finite_mask = np.isfinite(reference_x)
    mean_connectivity = np.nanmean(np.where(reference_finite_mask, reference_x, np.nan), axis=0)
    mean_connectivity = np.nan_to_num(mean_connectivity, nan=0.0, posinf=0.0, neginf=0.0)
    return np.where(finite_mask, x, mean_connectivity[np.newaxis, :, :])


def load_modality(name, atlas_path, label_path):
    print(f"loading ABIDE data {name}...")
    data = scio.loadmat(atlas_path)["connectivity"]
    labels = np.loadtxt(label_path).astype(np.int64)
    data = preprocess_connectivity(data, labels, normal_label=0)
    print("---------------------")
    print(f"X Atlas {name}:", data.shape)
    print(f"Y Atlas {name}:", labels.shape)
    print("---------------------")
    return data, labels


def save_tsne_plot(embeddings, labels, output_path, title):
    if embeddings.shape[0] < 3:
        return

    perplexity = min(30, embeddings.shape[0] - 1)
    perplexity = max(2, perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = sorted(np.unique(labels))
    label_name = {0: "HC", 1: "ASD"}
    colors = {0: "tab:blue", 1: "tab:orange"}
    for label_value in unique_labels:
        idx = labels == label_value
        ax.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            s=20,
            alpha=0.8,
            c=colors.get(int(label_value), "tab:gray"),
            label=label_name.get(int(label_value), f"label_{int(label_value)}"),
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_fold_audit(
    output_root,
    fold_index,
    labels,
    sim_weights,
    diff_weights,
    similarity_embeddings,
    difference_embeddings,
    modality_names,
):
    fold_dir = os.path.join(output_root, f"fold_{fold_index:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    labels = np.asarray(labels)
    sim_weights = np.asarray(sim_weights)
    diff_weights = np.asarray(diff_weights)

    lines = ["group,branch,modality,mean_weight"]
    groups = [("all", None), ("ASD", 1), ("HC", 0)]
    for group_name, group_label in groups:
        if group_label is None:
            idx = np.arange(labels.shape[0])
        else:
            idx = np.where(labels == group_label)[0]
        if idx.size == 0:
            continue
        sim_mean = sim_weights[idx].mean(axis=0)
        diff_mean = diff_weights[idx].mean(axis=0)
        for modality_idx, modality_name in enumerate(modality_names):
            lines.append(f"{group_name},similarity,{modality_name},{sim_mean[modality_idx]:.8f}")
            lines.append(f"{group_name},difference,{modality_name},{diff_mean[modality_idx]:.8f}")

    with open(os.path.join(fold_dir, "modality_gate_weights.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    save_tsne_plot(
        similarity_embeddings,
        labels,
        os.path.join(fold_dir, "tsne_similarity.png"),
        f"Fold {fold_index} Similarity Branch t-SNE",
    )
    save_tsne_plot(
        difference_embeddings,
        labels,
        os.path.join(fold_dir, "tsne_difference.png"),
        f"Fold {fold_index} Difference Branch t-SNE",
    )


class MultiModalBrainModel(nn.Module):
    # 保持集成模型思想不变，融合层简化为稳定的模态拼接
    def __init__(self, models, fusion_dim, feature_dim, num_classes, model_dim, num_heads=2, num_layers=2, dropout=0.5):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_modalities = len(models)

        # 将每个模态输出作为一个token: [B, M, D]
        self.token_proj = nn.Linear(feature_dim, model_dim) if feature_dim != model_dim else nn.Identity()
        stacked_dim = model_dim * self.num_modalities

        self.similarity_encoder = nn.Sequential(
            nn.LayerNorm(stacked_dim),
            nn.Linear(stacked_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.difference_encoder = nn.Sequential(
            nn.LayerNorm(stacked_dim),
            nn.Linear(stacked_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_list, return_branch_outputs=False, return_gate_weights=False):
        model_outputs = [model(x) for model, x in zip(self.models, x_list)]
        # [B, M, feature_dim]
        modality_tokens = torch.stack(model_outputs, dim=1)
        modality_tokens = self.token_proj(modality_tokens)
        stacked_tokens = modality_tokens.reshape(modality_tokens.size(0), -1)
        uniform_weights = torch.full(
            (modality_tokens.size(0), self.num_modalities),
            fill_value=1.0 / self.num_modalities,
            device=modality_tokens.device,
            dtype=modality_tokens.dtype,
        )

        similarity = self.similarity_encoder(stacked_tokens)
        difference = self.difference_encoder(stacked_tokens)
        class_output = self.classifier(similarity)
        if return_branch_outputs and return_gate_weights:
            return (
                class_output,
                similarity,
                difference,
                model_outputs,
                uniform_weights,
                uniform_weights,
            )
        if return_branch_outputs:
            return class_output, similarity, difference, model_outputs
        if return_gate_weights:
            return class_output, similarity, difference, uniform_weights, uniform_weights
        return class_output, similarity, difference


def rbf_kernel_matrix(x, y, sigma):
    x_sq = (x * x).sum(dim=1, keepdim=True)
    y_sq = (y * y).sum(dim=1, keepdim=True)
    dist_sq = x_sq - 2.0 * torch.matmul(x, y.t()) + y_sq.t()
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.exp(-dist_sq / (2.0 * sigma * sigma))


def mmd_rbf(x, y, sigmas=(0.5, 1.0, 2.0, 4.0)):
    loss = 0.0
    for sigma in sigmas:
        k_xx = rbf_kernel_matrix(x, x, sigma)
        k_yy = rbf_kernel_matrix(y, y, sigma)
        k_xy = rbf_kernel_matrix(x, y, sigma)
        loss = loss + (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return torch.clamp(loss / len(sigmas), min=0.0)


def similarity_loss(output_list, alpha=0.8, sigmas=(0.5, 1.0, 2.0, 4.0)):
    # 用MMD做分布对齐，而不是强制向量点对点相同
    normalized_outputs = [F.normalize(out, p=2, dim=1) for out in output_list]
    loss = 0.0
    count = 0
    for i in range(len(normalized_outputs)):
        for j in range(i + 1, len(normalized_outputs)):
            loss = loss + mmd_rbf(normalized_outputs[i], normalized_outputs[j], sigmas=sigmas)
            count += 1
    return alpha * (loss / max(count, 1))


def center_kernel_matrix(kernel):
    n = kernel.size(0)
    eye = torch.eye(n, device=kernel.device, dtype=kernel.dtype)
    one = torch.ones((n, n), device=kernel.device, dtype=kernel.dtype) / n
    h = eye - one
    return h @ kernel @ h


def hsic_rbf(x, y, sigma=1.0):
    # Normalized HSIC: Tr(KH L H) / (||KH||_F * ||LH||_F)
    if x.size(0) <= 1:
        return x.new_tensor(0.0)

    kx = rbf_kernel_matrix(x, x, sigma)
    ky = rbf_kernel_matrix(y, y, sigma)
    kx = center_kernel_matrix(kx)
    ky = center_kernel_matrix(ky)
    numerator = torch.trace(kx @ ky)
    denominator = torch.sqrt(torch.trace(kx @ kx) * torch.trace(ky @ ky) + 1e-12)
    hsic = numerator / denominator
    return torch.clamp(hsic, min=0.0)


def difference_loss(output_list, beta=0.01, sigma=1.0):
    # 最小化HSIC，推动不同模态学到统计独立的互补信息
    normalized_outputs = [F.normalize(out, p=2, dim=1) for out in output_list]
    loss = 0.0
    count = 0
    for i in range(len(normalized_outputs)):
        for j in range(i + 1, len(normalized_outputs)):
            loss = loss + hsic_rbf(normalized_outputs[i], normalized_outputs[j], sigma=sigma)
            count += 1
    return beta * (loss / max(count, 1))


class UncertaintyWeighting(nn.Module):
    # Keep classification as the primary objective, adapt only sim/diff terms
    # L = L_cls + exp(-s1)L_sim + exp(-s2)L_diff + 0.5(s1+s2), where s=log(sigma^2)
    def __init__(self, num_losses=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, class_loss, sim_loss, diff_loss):
        log_vars = torch.clamp(self.log_vars, min=-5.0, max=5.0)
        sim_term = torch.exp(-log_vars[0]) * sim_loss + 0.5 * log_vars[0]
        diff_term = torch.exp(-log_vars[1]) * diff_loss + 0.5 * log_vars[1]
        return class_loss + sim_term + diff_term


def find_best_threshold(y_true, probs, num_thresholds=81):
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    best_threshold = 0.5
    best_acc = -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(np.int64)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(thr)
    return best_threshold


setup_seed(123)
time_data_load_start = time.time()

# 当前启用的三个模态：AAL + CC400 + EZ
X1, Y1 = load_modality(
    name="aal",
    atlas_path="./ABIDEdata/pcc_correlation_871_aal_.mat",
    label_path="./ABIDEdata/871_label_aal.txt",
)
X3, Y3 = load_modality(
    name="cc400",
    atlas_path="./ABIDEdata/pcc_correlation_871_cc400_.mat",
    label_path="./ABIDEdata/871_label_cc400.txt",
)
X5, Y5 = load_modality(
    name="ez",
    atlas_path="./ABIDEdata/pcc_correlation_871_ez_.mat",
    label_path="./ABIDEdata/871_label_ez.txt",
)

if not (np.array_equal(Y1, Y3) and np.array_equal(Y1, Y5)):
    raise ValueError("Labels are not aligned across modalities.")

time_data_load_end = time.time()
Y = Y1

epochs = 40
batch_size = 64
grad_accum_steps = 2
dropout = 0.25
lr = 1e-4
logvar_lr = 1e-3
decay = 0.01
outer_folds = 10
val_ratio = 0.1
early_stop_patience = 10
early_stop_min_delta = 1e-4

NUM_HEADS_ez = 2
NUM_LAYERS_ez = 1
NUM_HEADS_aal = 2
NUM_LAYERS_aal = 1
NUM_HEADS_cc400 = 4
NUM_LAYERS_cc400 = 1
NUM_HEADS_multi = 8
NUM_LAYERS_multi = 3

model_dim = 64
feature_dim = 64
fusion_dim = 64
alpha = 0.1
beta = 0.2
mmd_sigmas = (0.5, 1.0, 2.0, 4.0)
hsic_sigma = 1.0
modality_names = ["aal", "cc400", "ez"]
audit_output_root = "./modelstfensemble8/audit"

result = []
recall_k = []
f1_k = []
auc_k = []
result_final = []
recall_list = []
f1_list = []
auc_list = []
acc_final = 0.0
time_train_list = []


for run_idx in range(1):
    setup_seed(run_idx)
    acc_all = 0.0
    kf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    for kfold_index, (trainval_index, test_index) in enumerate(kf.split(X1, Y), start=1):
        time_train_start = time.time()
        print("kfold_index:", kfold_index)

        train_index, val_index = train_test_split(
            trainval_index,
            test_size=val_ratio,
            random_state=42 + kfold_index,
            stratify=Y[trainval_index],
        )

        X1_train, X1_val, X1_test = X1[train_index], X1[val_index], X1[test_index]
        X3_train, X3_val, X3_test = X3[train_index], X3[val_index], X3[test_index]
        X5_train, X5_val, X5_test = X5[train_index], X5[val_index], X5[test_index]
        Y_train, Y_val, Y_test = Y[train_index], Y[val_index], Y[test_index]

        print("X1_train", X1_train.shape)
        print("X1_val", X1_val.shape)
        print("X1_test", X1_test.shape)
        print("Y_train", Y_train.shape)
        print("Y_val", Y_val.shape)
        print("Y_test", Y_test.shape)
        print("Positive samples in training set:", int(np.sum(Y_train == 1)))
        print("Negative samples in training set:", int(np.sum(Y_train == 0)))
        print("Positive samples in test set:", int(np.sum(Y_test == 1)))
        print("Negative samples in test set:", int(np.sum(Y_test == 0)))

        model_aal = SimpleTransformerModel(
            input_dim=X1.shape[-1],
            model_dim=model_dim,
            num_classes=feature_dim,
            num_heads=NUM_HEADS_aal,
            num_layers=NUM_LAYERS_aal,
            dropout=dropout,
        )
        model_cc400 = SimpleTransformerModel(
            input_dim=X3.shape[-1],
            model_dim=model_dim,
            num_classes=feature_dim,
            num_heads=NUM_HEADS_cc400,
            num_layers=NUM_LAYERS_cc400,
            dropout=dropout,
        )
        model_ez = SimpleTransformerModel(
            input_dim=X5.shape[-1],
            model_dim=model_dim,
            num_classes=feature_dim,
            num_heads=NUM_HEADS_ez,
            num_layers=NUM_LAYERS_ez,
            dropout=dropout,
        )

        models = [model_aal, model_cc400, model_ez]
        multi_modal_model = MultiModalBrainModel(
            models=models,
            fusion_dim=fusion_dim,
            feature_dim=feature_dim,
            num_classes=2,
            model_dim=model_dim,
            num_heads=NUM_HEADS_multi,
            num_layers=NUM_LAYERS_multi,
            dropout=dropout,
        ).to(device)

        loss_weighting = UncertaintyWeighting(num_losses=2).to(device)
        optimizer = optim.AdamW(
            [
                {"params": multi_modal_model.parameters(), "lr": lr, "weight_decay": decay},
                {"params": loss_weighting.parameters(), "lr": logvar_lr, "weight_decay": 0.0},
            ]
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state_dict = copy.deepcopy(multi_modal_model.state_dict())
        best_loss_weighting_state_dict = copy.deepcopy(loss_weighting.state_dict())
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            multi_modal_model.train()
            idx_batch = np.random.permutation(X1_train.shape[0])
            batch_indices = [
                idx_batch[start : start + batch_size]
                for start in range(0, X1_train.shape[0], batch_size)
            ]
            # BatchNorm1d在训练时要求batch size > 1，避免最后一个batch只有1个样本
            if len(batch_indices) > 1 and batch_indices[-1].shape[0] == 1:
                batch_indices[-2] = np.concatenate([batch_indices[-2], batch_indices[-1]])
                batch_indices = batch_indices[:-1]

            total_loss_list = []
            class_loss_list = []
            sim_loss_raw_list = []
            diff_loss_raw_list = []
            effective_step = 0
            optimizer.zero_grad()

            for batch in batch_indices:
                if batch.shape[0] <= 1:
                    continue

                effective_step += 1
                x1_batch = torch.from_numpy(X1_train[batch]).float().to(device)
                x3_batch = torch.from_numpy(X3_train[batch]).float().to(device)
                x5_batch = torch.from_numpy(X5_train[batch]).float().to(device)
                y_batch = torch.from_numpy(Y_train[batch]).long().to(device)

                outputs, _, _, branch_outputs = multi_modal_model(
                    [x1_batch, x3_batch, x5_batch], return_branch_outputs=True
                )
                output_aal, output_cc400, output_ez = branch_outputs

                class_loss = loss_fn(outputs, y_batch)
                sim_loss_raw = similarity_loss(
                    [output_aal, output_cc400, output_ez],
                    alpha=alpha,
                    sigmas=mmd_sigmas,
                )
                diff_loss_raw = difference_loss(
                    [output_aal, output_cc400, output_ez],
                    beta=beta,
                    sigma=hsic_sigma,
                )
                total_loss = loss_weighting(class_loss, sim_loss_raw, diff_loss_raw)

                (total_loss / grad_accum_steps).backward()
                if effective_step % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(multi_modal_model.parameters()) + list(loss_weighting.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss_list.append(total_loss.item())
                class_loss_list.append(class_loss.item())
                sim_loss_raw_list.append(sim_loss_raw.item())
                diff_loss_raw_list.append(diff_loss_raw.item())

            if effective_step > 0 and effective_step % grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    list(multi_modal_model.parameters()) + list(loss_weighting.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

            train_loss = float(np.mean(total_loss_list)) if total_loss_list else 0.0
            train_class_loss = float(np.mean(class_loss_list)) if class_loss_list else 0.0
            train_sim_loss = float(np.mean(sim_loss_raw_list)) if sim_loss_raw_list else 0.0
            train_diff_loss = float(np.mean(diff_loss_raw_list)) if diff_loss_raw_list else 0.0

            multi_modal_model.eval()
            with torch.no_grad():
                x1_val = torch.from_numpy(X1_val).float().to(device)
                x3_val = torch.from_numpy(X3_val).float().to(device)
                x5_val = torch.from_numpy(X5_val).float().to(device)
                y_val = torch.from_numpy(Y_val).long().to(device)

                val_outputs, _, _, val_branch_outputs = multi_modal_model(
                    [x1_val, x3_val, x5_val], return_branch_outputs=True
                )
                val_output_aal, val_output_cc400, val_output_ez = val_branch_outputs

                val_class_loss = loss_fn(val_outputs, y_val)
                val_sim_loss_raw = similarity_loss(
                    [val_output_aal, val_output_cc400, val_output_ez],
                    alpha=alpha,
                    sigmas=mmd_sigmas,
                )
                val_diff_loss_raw = difference_loss(
                    [val_output_aal, val_output_cc400, val_output_ez],
                    beta=beta,
                    sigma=hsic_sigma,
                )
                val_total_loss = loss_weighting(val_class_loss, val_sim_loss_raw, val_diff_loss_raw)

                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(Y_val, val_preds)

            if epoch % 5 == 0 or epoch == 1:
                sigma_vals = torch.exp(0.5 * loss_weighting.log_vars.detach()).cpu().numpy()
                print(
                    "epoch:",
                    epoch,
                    "train loss:",
                    train_loss,
                    "class_loss:",
                    train_class_loss,
                    "sim loss:",
                    train_sim_loss,
                    "diff loss:",
                    train_diff_loss,
                    "val loss:",
                    val_total_loss.item(),
                    "val sim:",
                    val_sim_loss_raw.item(),
                    "val diff:",
                    val_diff_loss_raw.item(),
                    "val acc:",
                    val_acc,
                    "sigma_sim:",
                    float(sigma_vals[0]),
                    "sigma_diff:",
                    float(sigma_vals[1]),
                )

            if val_total_loss.item() < best_val_loss - early_stop_min_delta:
                best_val_loss = val_total_loss.item()
                best_state_dict = copy.deepcopy(multi_modal_model.state_dict())
                best_loss_weighting_state_dict = copy.deepcopy(loss_weighting.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                    break

            scheduler.step()

        multi_modal_model.load_state_dict(best_state_dict)
        loss_weighting.load_state_dict(best_loss_weighting_state_dict)
        multi_modal_model.eval()
        with torch.no_grad():
            x1_val = torch.from_numpy(X1_val).float().to(device)
            x3_val = torch.from_numpy(X3_val).float().to(device)
            x5_val = torch.from_numpy(X5_val).float().to(device)
            val_outputs, _, _ = multi_modal_model([x1_val, x3_val, x5_val])
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            best_threshold = find_best_threshold(Y_val, val_probs)

            x1_test = torch.from_numpy(X1_test).float().to(device)
            x3_test = torch.from_numpy(X3_test).float().to(device)
            x5_test = torch.from_numpy(X5_test).float().to(device)
            test_outputs, similarity_test, difference_test, sim_gate_test, diff_gate_test = multi_modal_model(
                [x1_test, x3_test, x5_test], return_gate_weights=True
            )
            probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()

        preds = (probs >= best_threshold).astype(np.int64)
        acc = accuracy_score(Y_test, preds)
        recall = recall_score(Y_test, preds, average="macro", zero_division=1)
        f1 = f1_score(Y_test, preds, average="macro", zero_division=1)
        auc_score = roc_auc_score(Y_test, probs) if len(np.unique(Y_test)) > 1 else 0.5

        print(
            "Test acc",
            acc,
            "Test recall",
            recall,
            "Test f1",
            f1,
            "Test auc",
            auc_score,
            "best_thr",
            best_threshold,
        )

        os.makedirs("./modelstfensemble8", exist_ok=True)
        torch.save(multi_modal_model.state_dict(), f"./modelstfensemble8/{kfold_index}.pt")
        os.makedirs(audit_output_root, exist_ok=True)
        save_fold_audit(
            output_root=audit_output_root,
            fold_index=kfold_index,
            labels=Y_test,
            sim_weights=sim_gate_test.cpu().numpy(),
            diff_weights=diff_gate_test.cpu().numpy(),
            similarity_embeddings=similarity_test.cpu().numpy(),
            difference_embeddings=difference_test.cpu().numpy(),
            modality_names=modality_names,
        )
        sim_mean = sim_gate_test.mean(dim=0).cpu().numpy()
        diff_mean = diff_gate_test.mean(dim=0).cpu().numpy()
        print(
            "gate_mean(sim):",
            {name: float(weight) for name, weight in zip(modality_names, sim_mean)},
            "gate_mean(diff):",
            {name: float(weight) for name, weight in zip(modality_names, diff_mean)},
        )

        result.append([kfold_index, acc])
        recall_k.append([kfold_index, recall])
        f1_k.append([kfold_index, f1])
        auc_k.append([kfold_index, auc_score])
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc_score)
        acc_all += acc

        time_train_end = time.time()
        time_train_list.append(time_train_end - time_train_start)

    temp = acc_all / outer_folds
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / len(result_final)
    print("acc", result)
    print("recall", recall_k)
    print("f1", f1_k)
    print("AUC", auc_k)

print(result_final)
print(acc_final)
print(f"Ave Recall: {np.mean(recall_list)}")
print(f"Ave F1: {np.mean(f1_list)}")
print(f"Ave AUC: {np.mean(auc_list)}")
print(f"Data Loading Time: {time_data_load_end - time_data_load_start} seconds")
print(f"Ave Training Time: {np.mean(time_train_list)} seconds")
