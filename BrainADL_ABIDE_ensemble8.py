import copy
import glob
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


def validate_subject_alignment(modality_names, data_dir="./ABIDEdata"):
    subject_ids = {}
    candidate_templates = [
        "871_subject_ids_{name}.txt",
        "871_subject_id_{name}.txt",
        "{name}_subject_ids.txt",
        "{name}_subject_id.txt",
    ]

    for name in modality_names:
        candidates = [
            os.path.join(data_dir, template.format(name=name))
            for template in candidate_templates
        ]
        if not any(os.path.exists(path) for path in candidates):
            pattern_matches = glob.glob(os.path.join(data_dir, f"*subject*{name}*.txt"))
            candidates.extend(sorted(pattern_matches))

        for path in candidates:
            if os.path.exists(path):
                subject_ids[name] = np.loadtxt(path, dtype=str)
                break

    if len(subject_ids) != len(modality_names):
        missing = [name for name in modality_names if name not in subject_ids]
        print(
            "Subject ID files not found for",
            missing,
            "- relying on sample order alignment across modalities.",
        )
        return

    reference_name = modality_names[0]
    reference_ids = subject_ids[reference_name]
    for name in modality_names[1:]:
        if not np.array_equal(reference_ids, subject_ids[name]):
            raise ValueError(f"Subject IDs are not aligned between {reference_name} and {name}.")
    print("Subject IDs are aligned across modalities.")


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
    shared_weights,
    private_weights,
    shared_embeddings,
    private_embeddings,
    modality_names,
):
    fold_dir = os.path.join(output_root, f"fold_{fold_index:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    labels = np.asarray(labels)
    shared_weights = np.asarray(shared_weights)
    private_weights = np.asarray(private_weights)

    lines = ["group,branch,modality,mean_weight"]
    groups = [("all", None), ("ASD", 1), ("HC", 0)]
    for group_name, group_label in groups:
        if group_label is None:
            idx = np.arange(labels.shape[0])
        else:
            idx = np.where(labels == group_label)[0]
        if idx.size == 0:
            continue
        shared_mean = shared_weights[idx].mean(axis=0)
        private_mean = private_weights[idx].mean(axis=0)
        for modality_idx, modality_name in enumerate(modality_names):
            lines.append(f"{group_name},shared,{modality_name},{shared_mean[modality_idx]:.8f}")
            lines.append(f"{group_name},private,{modality_name},{private_mean[modality_idx]:.8f}")

    with open(os.path.join(fold_dir, "modality_gate_weights.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    save_tsne_plot(
        shared_embeddings,
        labels,
        os.path.join(fold_dir, "tsne_shared.png"),
        f"Fold {fold_index} Shared Branch t-SNE",
    )
    save_tsne_plot(
        private_embeddings,
        labels,
        os.path.join(fold_dir, "tsne_private.png"),
        f"Fold {fold_index} Private Branch t-SNE",
    )


class BranchProjector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalBrainModel(nn.Module):
    def __init__(self, models, fusion_dim, feature_dim, num_classes, dropout=0.5):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_modalities = len(models)

        self.shared_heads = nn.ModuleList(
            [BranchProjector(feature_dim, fusion_dim, dropout) for _ in range(self.num_modalities)]
        )
        self.private_heads = nn.ModuleList(
            [BranchProjector(feature_dim, fusion_dim, dropout) for _ in range(self.num_modalities)]
        )
        self.shared_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 1),
        )
        self.private_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 1),
        )
        self.aux_classifiers = nn.ModuleList(
            [nn.Linear(fusion_dim * 2, num_classes) for _ in range(self.num_modalities)]
        )

        classifier_input_dim = fusion_dim + fusion_dim * self.num_modalities
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def _compute_gate_weights(self, branch_outputs, gate_module):
        gate_scores = torch.cat([gate_module(branch_output) for branch_output in branch_outputs], dim=1)
        return torch.softmax(gate_scores, dim=1)

    def forward(self, x_list, return_details=False):
        branch_outputs = [model(x) for model, x in zip(self.models, x_list)]
        shared_outputs = [
            projector(branch_output)
            for projector, branch_output in zip(self.shared_heads, branch_outputs)
        ]
        private_outputs = [
            projector(branch_output)
            for projector, branch_output in zip(self.private_heads, branch_outputs)
        ]

        shared_weights = self._compute_gate_weights(shared_outputs, self.shared_gate)
        private_weights = self._compute_gate_weights(private_outputs, self.private_gate)

        shared_stack = torch.stack(shared_outputs, dim=1)
        private_stack = torch.stack(private_outputs, dim=1)

        shared_fused = torch.sum(shared_stack * shared_weights.unsqueeze(-1), dim=1)
        private_weighted = private_stack * private_weights.unsqueeze(-1)
        private_fused = private_weighted.reshape(private_weighted.size(0), -1)

        logits = self.classifier(torch.cat([shared_fused, private_fused], dim=1))

        if not return_details:
            return logits

        aux_logits = [
            aux_classifier(torch.cat([shared_output, private_output], dim=1))
            for aux_classifier, shared_output, private_output in zip(
                self.aux_classifiers, shared_outputs, private_outputs
            )
        ]
        details = {
            "branch_outputs": branch_outputs,
            "shared_outputs": shared_outputs,
            "private_outputs": private_outputs,
            "shared_fused": shared_fused,
            "private_fused": private_fused,
            "shared_weights": shared_weights,
            "private_weights": private_weights,
            "aux_logits": aux_logits,
        }
        return logits, details


def paired_cosine_alignment_loss(shared_outputs, weight=0.1):
    normalized_outputs = [F.normalize(out, p=2, dim=1) for out in shared_outputs]
    loss = 0.0
    count = 0
    for i in range(len(normalized_outputs)):
        for j in range(i + 1, len(normalized_outputs)):
            cosine_sim = F.cosine_similarity(normalized_outputs[i], normalized_outputs[j], dim=1)
            loss = loss + (1.0 - cosine_sim.mean())
            count += 1
    return weight * (loss / max(count, 1))


def private_decorrelation_loss(shared_outputs, private_outputs, cross_weight=0.1, orth_weight=0.1):
    normalized_shared = [F.normalize(out, p=2, dim=1) for out in shared_outputs]
    normalized_private = [F.normalize(out, p=2, dim=1) for out in private_outputs]

    cross_modal_loss = 0.0
    cross_count = 0
    for i in range(len(normalized_private)):
        for j in range(i + 1, len(normalized_private)):
            cosine_sq = (normalized_private[i] * normalized_private[j]).sum(dim=1).pow(2).mean()
            cross_modal_loss = cross_modal_loss + cosine_sq
            cross_count += 1

    orth_loss = 0.0
    for shared_output, private_output in zip(normalized_shared, normalized_private):
        cosine_sq = (shared_output * private_output).sum(dim=1).pow(2).mean()
        orth_loss = orth_loss + cosine_sq

    cross_modal_loss = cross_modal_loss / max(cross_count, 1)
    orth_loss = orth_loss / max(len(normalized_private), 1)
    return cross_weight * cross_modal_loss + orth_weight * orth_loss


def auxiliary_branch_loss(aux_logits, targets, loss_fn, weight=0.1):
    if not aux_logits:
        return torch.zeros((), device=targets.device)
    loss = sum(loss_fn(aux_logit, targets) for aux_logit in aux_logits) / len(aux_logits)
    return weight * loss


def mean_feature_std(embeddings):
    if embeddings.size(0) <= 1:
        return 0.0
    return float(embeddings.std(dim=0).mean().item())


def collect_grad_norms(model):
    tracked_prefixes = {
        "shared_heads": "shared_heads",
        "private_heads": "private_heads",
        "shared_gate": "shared_gate",
        "private_gate": "private_gate",
        "classifier": "classifier",
        "aux_classifiers": "aux_classifiers",
    }
    grad_norms = {key: [] for key in tracked_prefixes}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        for key, prefix in tracked_prefixes.items():
            if name.startswith(prefix):
                grad_norms[key].append(param.grad.norm().item())
                break
    return {
        key: (float(np.mean(values)) if values else None)
        for key, values in grad_norms.items()
    }


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


def compute_binary_auc(y_true, probs):
    return roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5


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
validate_subject_alignment(["aal", "cc400", "ez"])

time_data_load_end = time.time()
Y = Y1

epochs = 40
batch_size = 64
grad_accum_steps = 2
dropout = 0.25
lr = 1e-4
decay = 0.01
outer_folds = 10
val_ratio = 0.1
early_stop_patience = 10
early_stop_min_delta = 1e-4
early_stop_auc_delta = 1e-4
early_stop_min_epochs = 15
early_stop_tie_auc_window = 1e-3

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
lambda_align = 0.1
lambda_private_cross = 0.1
lambda_private_orth = 0.1
lambda_aux = 0.05
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
            dropout=dropout,
        ).to(device)

        optimizer = optim.AdamW(multi_modal_model.parameters(), lr=lr, weight_decay=decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        loss_fn = nn.CrossEntropyLoss()

        best_val_auc = float("-inf")
        best_val_class_loss = float("inf")
        best_epoch = 0
        best_state_dict = copy.deepcopy(multi_modal_model.state_dict())
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            multi_modal_model.train()
            idx_batch = np.random.permutation(X1_train.shape[0])
            batch_indices = [
                idx_batch[start : start + batch_size]
                for start in range(0, X1_train.shape[0], batch_size)
            ]
            # 避免最后一个 batch 只有 1 个样本，减少统计量与损失波动
            if len(batch_indices) > 1 and batch_indices[-1].shape[0] == 1:
                batch_indices[-2] = np.concatenate([batch_indices[-2], batch_indices[-1]])
                batch_indices = batch_indices[:-1]

            total_loss_list = []
            class_loss_list = []
            align_loss_list = []
            private_loss_list = []
            aux_loss_list = []
            effective_step = 0
            grad_norm_snapshot = {}
            optimizer.zero_grad()

            for batch in batch_indices:
                if batch.shape[0] <= 1:
                    continue

                effective_step += 1
                x1_batch = torch.from_numpy(X1_train[batch]).float().to(device)
                x3_batch = torch.from_numpy(X3_train[batch]).float().to(device)
                x5_batch = torch.from_numpy(X5_train[batch]).float().to(device)
                y_batch = torch.from_numpy(Y_train[batch]).long().to(device)

                outputs, details = multi_modal_model([x1_batch, x3_batch, x5_batch], return_details=True)

                class_loss = loss_fn(outputs, y_batch)
                align_loss = paired_cosine_alignment_loss(
                    details["shared_outputs"],
                    weight=lambda_align,
                )
                private_loss = private_decorrelation_loss(
                    details["shared_outputs"],
                    details["private_outputs"],
                    cross_weight=lambda_private_cross,
                    orth_weight=lambda_private_orth,
                )
                aux_loss = auxiliary_branch_loss(
                    details["aux_logits"],
                    y_batch,
                    loss_fn,
                    weight=lambda_aux,
                )
                total_loss = class_loss + align_loss + private_loss + aux_loss

                (total_loss / grad_accum_steps).backward()
                if effective_step % grad_accum_steps == 0:
                    grad_norm_snapshot = collect_grad_norms(multi_modal_model)
                    torch.nn.utils.clip_grad_norm_(
                        multi_modal_model.parameters(),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss_list.append(total_loss.item())
                class_loss_list.append(class_loss.item())
                align_loss_list.append(align_loss.item())
                private_loss_list.append(private_loss.item())
                aux_loss_list.append(aux_loss.item())

            if effective_step > 0 and effective_step % grad_accum_steps != 0:
                grad_norm_snapshot = collect_grad_norms(multi_modal_model)
                torch.nn.utils.clip_grad_norm_(
                    multi_modal_model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

            train_loss = float(np.mean(total_loss_list)) if total_loss_list else 0.0
            train_class_loss = float(np.mean(class_loss_list)) if class_loss_list else 0.0
            train_align_loss = float(np.mean(align_loss_list)) if align_loss_list else 0.0
            train_private_loss = float(np.mean(private_loss_list)) if private_loss_list else 0.0
            train_aux_loss = float(np.mean(aux_loss_list)) if aux_loss_list else 0.0

            multi_modal_model.eval()
            with torch.no_grad():
                x1_val = torch.from_numpy(X1_val).float().to(device)
                x3_val = torch.from_numpy(X3_val).float().to(device)
                x5_val = torch.from_numpy(X5_val).float().to(device)
                y_val = torch.from_numpy(Y_val).long().to(device)

                val_outputs, val_details = multi_modal_model([x1_val, x3_val, x5_val], return_details=True)

                val_class_loss = loss_fn(val_outputs, y_val)
                val_align_loss = paired_cosine_alignment_loss(
                    val_details["shared_outputs"],
                    weight=lambda_align,
                )
                val_private_loss = private_decorrelation_loss(
                    val_details["shared_outputs"],
                    val_details["private_outputs"],
                    cross_weight=lambda_private_cross,
                    orth_weight=lambda_private_orth,
                )
                val_aux_loss = auxiliary_branch_loss(
                    val_details["aux_logits"],
                    y_val,
                    loss_fn,
                    weight=lambda_aux,
                )
                val_total_loss = val_class_loss + val_align_loss + val_private_loss + val_aux_loss

                val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_acc = accuracy_score(Y_val, val_preds)
                val_auc = compute_binary_auc(Y_val, val_probs)
                val_branch_std = {
                    name: mean_feature_std(branch_output)
                    for name, branch_output in zip(modality_names, val_details["branch_outputs"])
                }
                val_shared_std = mean_feature_std(val_details["shared_fused"])
                val_private_std = mean_feature_std(val_details["private_fused"])

            if epoch % 5 == 0 or epoch == 1:
                print(
                    "epoch:",
                    epoch,
                    "train loss:",
                    train_loss,
                    "class_loss:",
                    train_class_loss,
                    "align loss:",
                    train_align_loss,
                    "private loss:",
                    train_private_loss,
                    "aux loss:",
                    train_aux_loss,
                    "val loss:",
                    val_total_loss.item(),
                    "val class:",
                    val_class_loss.item(),
                    "val align:",
                    val_align_loss.item(),
                    "val private:",
                    val_private_loss.item(),
                    "val aux:",
                    val_aux_loss.item(),
                    "val acc:",
                    val_acc,
                    "val auc:",
                    val_auc,
                    "grad norms:",
                    grad_norm_snapshot,
                    "branch std:",
                    val_branch_std,
                    "shared std:",
                    val_shared_std,
                    "private std:",
                    val_private_std,
                )

            improved = False
            if val_auc > best_val_auc + early_stop_auc_delta:
                improved = True
            elif (
                abs(val_auc - best_val_auc) <= early_stop_tie_auc_window
                and val_class_loss.item() < best_val_class_loss - early_stop_min_delta
            ):
                improved = True

            if improved:
                best_val_auc = val_auc
                best_val_class_loss = val_class_loss.item()
                best_epoch = epoch
                best_state_dict = copy.deepcopy(multi_modal_model.state_dict())
                epochs_no_improve = 0
                print(
                    f"Checkpoint updated at epoch {epoch}: "
                    f"val_auc={best_val_auc:.6f}, val_class_loss={best_val_class_loss:.6f}"
                )
            else:
                if epoch >= early_stop_min_epochs:
                    epochs_no_improve += 1
                if epoch >= early_stop_min_epochs and epochs_no_improve >= early_stop_patience:
                    print(
                        f"Early stopping at epoch {epoch}, "
                        f"best epoch: {best_epoch}, "
                        f"best val auc: {best_val_auc:.6f}, "
                        f"best val class loss: {best_val_class_loss:.6f}"
                    )
                    break

            scheduler.step()

        multi_modal_model.load_state_dict(best_state_dict)
        multi_modal_model.eval()
        with torch.no_grad():
            x1_val = torch.from_numpy(X1_val).float().to(device)
            x3_val = torch.from_numpy(X3_val).float().to(device)
            x5_val = torch.from_numpy(X5_val).float().to(device)
            val_outputs = multi_modal_model([x1_val, x3_val, x5_val])
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            best_threshold = find_best_threshold(Y_val, val_probs)

            x1_test = torch.from_numpy(X1_test).float().to(device)
            x3_test = torch.from_numpy(X3_test).float().to(device)
            x5_test = torch.from_numpy(X5_test).float().to(device)
            test_outputs, test_details = multi_modal_model([x1_test, x3_test, x5_test], return_details=True)
            probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()

        preds = (probs >= best_threshold).astype(np.int64)
        acc = accuracy_score(Y_test, preds)
        recall = recall_score(Y_test, preds, average="macro", zero_division=1)
        f1 = f1_score(Y_test, preds, average="macro", zero_division=1)
        auc_score = compute_binary_auc(Y_test, probs)

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
            shared_weights=test_details["shared_weights"].cpu().numpy(),
            private_weights=test_details["private_weights"].cpu().numpy(),
            shared_embeddings=test_details["shared_fused"].cpu().numpy(),
            private_embeddings=test_details["private_fused"].cpu().numpy(),
            modality_names=modality_names,
        )
        shared_mean = test_details["shared_weights"].mean(dim=0).cpu().numpy()
        private_mean = test_details["private_weights"].mean(dim=0).cpu().numpy()
        print(
            "gate_mean(shared):",
            {name: float(weight) for name, weight in zip(modality_names, shared_mean)},
            "gate_mean(private):",
            {name: float(weight) for name, weight in zip(modality_names, private_mean)},
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
