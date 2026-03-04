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
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
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


class MultiModalBrainModel(nn.Module):
    # 保持集成模型结构不变
    def __init__(self, models, fusion_dim, feature_dim, num_classes, model_dim, num_heads=2, num_layers=2, dropout=0.5):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_modalities = len(models)

        # 将每个模态输出作为一个token: [B, M, D]
        self.token_proj = nn.Linear(feature_dim, model_dim) if feature_dim != model_dim else nn.Identity()

        sim_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        diff_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.similarity_transformer = nn.TransformerEncoder(sim_layer, num_layers=num_layers)
        self.difference_transformer = nn.TransformerEncoder(diff_layer, num_layers=num_layers)

        self.similarity_encoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, fusion_dim),
        )

        self.difference_encoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, fusion_dim),
        )

        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_list, return_branch_outputs=False):
        model_outputs = [model(x) for model, x in zip(self.models, x_list)]
        # [B, M, feature_dim]
        modality_tokens = torch.stack(model_outputs, dim=1)
        modality_tokens = self.token_proj(modality_tokens)

        sim_tokens = self.similarity_transformer(modality_tokens)
        diff_tokens = self.difference_transformer(modality_tokens)

        similarity = self.similarity_encoder(sim_tokens.mean(dim=1))
        difference = self.difference_encoder(diff_tokens.mean(dim=1))
        class_output = self.classifier(similarity)
        if return_branch_outputs:
            return class_output, similarity, difference, model_outputs
        return class_output, similarity, difference


def similarity_loss(output_list, alpha=0.5):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    normalized_outputs = [F.normalize(out, p=2, dim=1) for out in output_list]
    loss = 0.0
    n = len(normalized_outputs)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = cos(normalized_outputs[i], normalized_outputs[j])
            loss += (1 - sim).mean()
            count += 1
    return alpha * (loss / max(count, 1))


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive, negatives):
        # anchor: [B, D], positive: [B, D], negatives: [B, Nneg, D]
        anchor = F.normalize(anchor, p=2, dim=1).unsqueeze(1)  # [B, 1, D]
        positive = F.normalize(positive, p=2, dim=1).unsqueeze(1)  # [B, 1, D]
        if negatives.dim() == 2:
            negatives = negatives.unsqueeze(1)
        if negatives.dim() != 3:
            raise ValueError(f"negatives must be [B, Nneg, D], got shape {negatives.shape}")
        negatives = F.normalize(negatives, p=2, dim=2)

        pos_logits = self.cosine_similarity(anchor, positive) / self.temperature  # [B, 1]
        neg_logits = self.cosine_similarity(anchor, negatives) / self.temperature  # [B, Nneg]
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1+Nneg]
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, targets)
        return loss


def build_infonce_negatives(positive_modal, aux_modal):
    # 构造多负样本: 仅使用跨受试者负样本，避免与相似性损失直接冲突
    # 输出: [B, Nneg, D]
    if positive_modal.size(0) <= 1:
        return aux_modal.unsqueeze(1)
    neg1 = torch.roll(positive_modal, shifts=1, dims=0)
    neg2 = torch.roll(aux_modal, shifts=1, dims=0)
    return torch.stack([neg1, neg2], dim=1)


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
dropout = 0.25
lr = 1e-4
decay = 0.01
outer_folds = 10
val_ratio = 0.1
early_stop_patience = 10
early_stop_min_delta = 1e-4

NUM_HEADS_ez = 1
NUM_LAYERS_ez = 1
NUM_HEADS_aal = 1
NUM_LAYERS_aal = 1
NUM_HEADS_cc400 = 4
NUM_LAYERS_cc400 = 1
NUM_HEADS_multi = 8
NUM_LAYERS_multi = 3

model_dim = 64
feature_dim = 64
fusion_dim = 64
alpha = 0.6
beta = 0.008

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

        optimizer = optim.AdamW(multi_modal_model.parameters(), lr=lr, weight_decay=decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        loss_fn = nn.CrossEntropyLoss()
        infoNCE_loss_fn = InfoNCELoss()

        best_val_loss = float("inf")
        best_state_dict = copy.deepcopy(multi_modal_model.state_dict())
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
            sim_loss_list = []
            diff_loss_list = []

            for batch in batch_indices:
                if batch.shape[0] <= 1:
                    continue

                x1_batch = torch.from_numpy(X1_train[batch]).float().to(device)
                x3_batch = torch.from_numpy(X3_train[batch]).float().to(device)
                x5_batch = torch.from_numpy(X5_train[batch]).float().to(device)
                y_batch = torch.from_numpy(Y_train[batch]).long().to(device)

                optimizer.zero_grad()

                outputs, similarity_output, difference_output, branch_outputs = multi_modal_model(
                    [x1_batch, x3_batch, x5_batch], return_branch_outputs=True
                )
                output_aal, output_cc400, output_ez = branch_outputs

                class_loss = loss_fn(outputs, y_batch)
                reg_scale = min(1.0, epoch / 10.0)
                sim_loss = reg_scale * similarity_loss([output_aal, output_cc400, output_ez], alpha)
                neg_for_cc400 = build_infonce_negatives(output_ez, output_aal)
                neg_for_ez = build_infonce_negatives(output_cc400, output_aal)
                contrast_loss = reg_scale * beta * 0.5 * (
                    infoNCE_loss_fn(output_cc400, output_ez, neg_for_cc400)
                    + infoNCE_loss_fn(output_ez, output_cc400, neg_for_ez)
                )
                total_loss = class_loss + sim_loss + contrast_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(multi_modal_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss_list.append(total_loss.item())
                class_loss_list.append(class_loss.item())
                sim_loss_list.append(sim_loss.item())
                diff_loss_list.append(contrast_loss.item())

            train_loss = float(np.mean(total_loss_list)) if total_loss_list else 0.0
            train_class_loss = float(np.mean(class_loss_list)) if class_loss_list else 0.0
            train_sim_loss = float(np.mean(sim_loss_list)) if sim_loss_list else 0.0
            train_diff_loss = float(np.mean(diff_loss_list)) if diff_loss_list else 0.0

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
                reg_scale = min(1.0, epoch / 10.0)
                val_sim_loss = reg_scale * similarity_loss([val_output_aal, val_output_cc400, val_output_ez], alpha)
                val_neg_for_cc400 = build_infonce_negatives(val_output_ez, val_output_aal)
                val_neg_for_ez = build_infonce_negatives(val_output_cc400, val_output_aal)
                val_contrast_loss = reg_scale * beta * 0.5 * (
                    infoNCE_loss_fn(val_output_cc400, val_output_ez, val_neg_for_cc400)
                    + infoNCE_loss_fn(val_output_ez, val_output_cc400, val_neg_for_ez)
                )
                val_total_loss = val_class_loss + val_sim_loss + val_contrast_loss

                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(Y_val, val_preds)

            if epoch % 5 == 0 or epoch == 1:
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
                    "val acc:",
                    val_acc,
                )

            if val_total_loss.item() < best_val_loss - early_stop_min_delta:
                best_val_loss = val_total_loss.item()
                best_state_dict = copy.deepcopy(multi_modal_model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                    break

            scheduler.step()

        multi_modal_model.load_state_dict(best_state_dict)
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
            test_outputs, _, _ = multi_modal_model([x1_test, x3_test, x5_test])
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
