import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from torch.optim import lr_scheduler

try:
    import seaborn as sns
except ImportError:
    sns = None

load_dotenv()
device = torch.device(os.getenv("DEVICE", "cpu"))
print(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_attention_rollout(attn_stack):
    # attn_stack: [batch, num_layers, num_heads, num_nodes, num_nodes]
    attn = attn_stack.mean(dim=2)
    num_nodes = attn.size(-1)
    eye = torch.eye(num_nodes, device=attn.device, dtype=attn.dtype).unsqueeze(0).unsqueeze(0)
    attn = attn + eye
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    rollout = attn[:, 0]
    for layer_idx in range(1, attn.size(1)):
        rollout = torch.bmm(attn[:, layer_idx], rollout)
    return rollout


def save_attention_heatmap(attn_matrix, save_path, title, cmap="viridis", center=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9, 8))
    if sns is not None:
        sns.heatmap(
            attn_matrix,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            xticklabels=20,
            yticklabels=20,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
    else:
        im = ax.imshow(attn_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(np.arange(0, attn_matrix.shape[1], 20))
        ax.set_yticks(np.arange(0, attn_matrix.shape[0], 20))

    ax.set_title(title)
    ax.set_xlabel("Key Nodes")
    ax.set_ylabel("Query Nodes")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def save_top_nodes(attn_matrix, save_path, node_labels, top_k=10, use_abs=False):
    if use_abs:
        scores = np.abs(attn_matrix).sum(axis=1)
    else:
        scores = attn_matrix.sum(axis=1)
    top_indices = np.argsort(scores)[::-1][:top_k]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("rank,node_index,node_label,score\n")
        for rank, idx in enumerate(top_indices, start=1):
            f.write(f"{rank},{idx},{node_labels[idx]},{scores[idx]:.8f}\n")


def export_fold_attention_maps(
    attn_last_np,
    attn_rollout_np,
    labels,
    fold_dir,
    node_labels,
    asd_label=1,
    hc_label=0,
    top_k=10,
):
    os.makedirs(fold_dir, exist_ok=True)

    for label_value, label_name in [(asd_label, "ASD"), (hc_label, "HC")]:
        sample_indices = np.where(labels == label_value)[0]
        if sample_indices.size == 0:
            continue
        sample_idx = int(sample_indices[0])
        save_attention_heatmap(
            attn_last_np[sample_idx],
            os.path.join(fold_dir, f"{label_name.lower()}_sample_last_layer.png"),
            f"Fold Sample ({label_name}) - Last Layer Head Mean Attention",
            cmap="viridis",
        )
        save_attention_heatmap(
            attn_rollout_np[sample_idx],
            os.path.join(fold_dir, f"{label_name.lower()}_sample_rollout.png"),
            f"Fold Sample ({label_name}) - Attention Rollout",
            cmap="viridis",
        )

    asd_idx = np.where(labels == asd_label)[0]
    hc_idx = np.where(labels == hc_label)[0]
    if asd_idx.size == 0 or hc_idx.size == 0:
        return

    asd_last_mean = attn_last_np[asd_idx].mean(axis=0)
    hc_last_mean = attn_last_np[hc_idx].mean(axis=0)
    diff_last = asd_last_mean - hc_last_mean
    diff_last_bound = float(np.max(np.abs(diff_last)))

    save_attention_heatmap(
        asd_last_mean,
        os.path.join(fold_dir, "asd_mean_last_layer.png"),
        "ASD Mean Attention - Last Layer Head Mean",
        cmap="viridis",
    )
    save_attention_heatmap(
        hc_last_mean,
        os.path.join(fold_dir, "hc_mean_last_layer.png"),
        "HC Mean Attention - Last Layer Head Mean",
        cmap="viridis",
    )
    save_attention_heatmap(
        diff_last,
        os.path.join(fold_dir, "asd_minus_hc_last_layer.png"),
        "ASD - HC Attention Difference (Last Layer Head Mean)",
        cmap="coolwarm",
        center=0.0,
        vmin=-diff_last_bound,
        vmax=diff_last_bound,
    )

    asd_rollout_mean = attn_rollout_np[asd_idx].mean(axis=0)
    hc_rollout_mean = attn_rollout_np[hc_idx].mean(axis=0)
    diff_rollout = asd_rollout_mean - hc_rollout_mean
    diff_rollout_bound = float(np.max(np.abs(diff_rollout)))

    save_attention_heatmap(
        asd_rollout_mean,
        os.path.join(fold_dir, "asd_mean_rollout.png"),
        "ASD Mean Attention - Rollout",
        cmap="viridis",
    )
    save_attention_heatmap(
        hc_rollout_mean,
        os.path.join(fold_dir, "hc_mean_rollout.png"),
        "HC Mean Attention - Rollout",
        cmap="viridis",
    )
    save_attention_heatmap(
        diff_rollout,
        os.path.join(fold_dir, "asd_minus_hc_rollout.png"),
        "ASD - HC Attention Difference (Rollout)",
        cmap="coolwarm",
        center=0.0,
        vmin=-diff_rollout_bound,
        vmax=diff_rollout_bound,
    )

    save_top_nodes(
        asd_last_mean,
        os.path.join(fold_dir, "top10_busy_nodes_asd_last_layer.csv"),
        node_labels,
        top_k=top_k,
        use_abs=False,
    )
    save_top_nodes(
        hc_last_mean,
        os.path.join(fold_dir, "top10_busy_nodes_hc_last_layer.csv"),
        node_labels,
        top_k=top_k,
        use_abs=False,
    )
    save_top_nodes(
        diff_last,
        os.path.join(fold_dir, "top10_diff_nodes_abs_last_layer.csv"),
        node_labels,
        top_k=top_k,
        use_abs=True,
    )


class AttnTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attn_weights = attn_weights.detach()
        return self.dropout1(x)


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_heads=4, num_layers=2, dropout=0.25):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_dim, model_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        transformer_layer = AttnTransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_fc = nn.Linear(model_dim, num_classes)

    def _collect_attn_stack(self):
        all_layers = []
        for layer in self.transformer.layers:
            if layer.last_attn_weights is not None:
                all_layers.append(layer.last_attn_weights)
        if not all_layers:
            return None
        return torch.stack(all_layers, dim=1)

    def forward(self, x, return_attention=False, use_rollout=False):
        # x: [batch, 200, 200], token是脑区，feature是该脑区到其他脑区的连接模式
        x = self.input_fc(x)
        x = x + self.pos_embedding[:, :x.size(1), :]

        for layer in self.transformer.layers:
            layer.last_attn_weights = None
        x = self.transformer(x)

        x = self.output_norm(x.mean(dim=1))
        logits = self.output_fc(x)
        if not return_attention:
            return logits

        attn_stack = self._collect_attn_stack()
        if attn_stack is None:
            if use_rollout:
                return logits, None, None
            return logits, None

        last_layer_head_mean = attn_stack[:, -1].mean(dim=1)
        if use_rollout:
            rollout = compute_attention_rollout(attn_stack)
            return logits, last_layer_head_mean, rollout
        return logits, last_layer_head_mean


setup_seed(123)

print("loading ABIDE data...")
Data_atlas1 = scio.loadmat("./ABIDEdata/pcc_correlation_871_cc200_.mat")
X = Data_atlas1["connectivity"]
Y = np.loadtxt("./ABIDEdata/871_label_cc200.txt").astype(np.int64)

Nodes_Atlas1 = X.shape[-1]

finite_mask = np.isfinite(X)
normal_subject_mask = Y == 0
reference_X = X[normal_subject_mask] if np.any(normal_subject_mask) else X
reference_finite_mask = np.isfinite(reference_X)
mean_connectivity = np.nanmean(np.where(reference_finite_mask, reference_X, np.nan), axis=0)
mean_connectivity = np.nan_to_num(mean_connectivity, nan=0.0, posinf=0.0, neginf=0.0)
X = np.where(finite_mask, X, mean_connectivity[np.newaxis, :, :])

print("---------------------")
print("X Atlas1:", X.shape)
print("Y Atlas1:", Y.shape)
print("---------------------")

epochs = 30
batch_size = 64
dropout = 0.25
lr = 1e-4
decay = 0.01
outer_folds = 10
val_ratio = 0.1
early_stop_patience = 10
early_stop_min_delta = 1e-4
result = []
acc_final = 0
result_final = []
model_dim = 64
input_dim = Nodes_Atlas1

attention_output_root = "./modelstf/attention_maps"
os.makedirs(attention_output_root, exist_ok=True)
asd_label = 1
hc_label = 0
top_k_nodes = 10
node_labels = [f"Node_{i:03d}" for i in range(Nodes_Atlas1)]

time_train_list = []
for run_idx in range(1):
    setup_seed(run_idx)

    nodes_number = Nodes_Atlas1
    nums = np.ones(Nodes_Atlas1)
    nums[: Nodes_Atlas1 - nodes_number] = 0
    np.random.seed(1)
    np.random.shuffle(nums)
    mask = nums.reshape(nums.shape[0], 1) * nums
    masked_X = X * mask
    j_bound = nodes_number
    for i in range(0, j_bound):
        idx = i
        if nums[idx] == 0:
            for j in range(j_bound, Nodes_Atlas1):
                if nums[j] == 1:
                    masked_X[:, [idx, j], :] = masked_X[:, [j, idx], :]
                    masked_X[:, :, [idx, j]] = masked_X[:, :, [j, idx]]
                    j_bound = j + 1
                    break

    masked_X = masked_X[:, :nodes_number, :nodes_number]
    X = masked_X

    global_last_sum = {
        asd_label: np.zeros((nodes_number, nodes_number), dtype=np.float64),
        hc_label: np.zeros((nodes_number, nodes_number), dtype=np.float64),
    }
    global_rollout_sum = {
        asd_label: np.zeros((nodes_number, nodes_number), dtype=np.float64),
        hc_label: np.zeros((nodes_number, nodes_number), dtype=np.float64),
    }
    global_count = {asd_label: 0, hc_label: 0}

    acc_all = 0
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    for kfold_index, (trainval_index, test_index) in enumerate(kf.split(X, Y), start=1):
        time_train_start = time.time()
        print("kfold_index:", kfold_index)

        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_trainval,
            Y_trainval,
            test_size=val_ratio,
            random_state=42 + kfold_index,
            stratify=Y_trainval,
        )

        print("X_train", X_train.shape)
        print("X_val", X_val.shape)
        print("X_test", X_test.shape)
        print("Y_train", Y_train.shape)
        print("Y_val", Y_val.shape)
        print("Y_test", Y_test.shape)

        model = SimpleTransformerModel(input_dim, model_dim, num_classes=2, dropout=dropout)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        loss_fn = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()

            idx_batch = np.random.permutation(int(X_train.shape[0]))
            loss_train_list = []
            for start in range(0, X_train.shape[0], int(batch_size)):
                batch = idx_batch[start : start + int(batch_size)]
                train_data_batch = X_train[batch]
                train_label_batch = Y_train[batch]
                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                optimizer.zero_grad()
                outputs = model(train_data_batch_dev)
                loss = loss_fn(outputs, train_label_batch_dev)
                loss_train_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            loss_train = float(np.mean(loss_train_list)) if loss_train_list else 0.0

            model.eval()
            with torch.no_grad():
                val_data_batch_dev = torch.from_numpy(X_val).float().to(device)
                val_label_batch_dev = torch.from_numpy(Y_val).long().to(device)
                val_outputs = model(val_data_batch_dev)
                val_loss = loss_fn(val_outputs, val_label_batch_dev).item()
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_acc = metrics.accuracy_score(Y_val, val_preds)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    "epoch:",
                    epoch,
                    "train loss:",
                    loss_train,
                    "val loss:",
                    val_loss,
                    "val acc:",
                    val_acc,
                )

            if val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                    break

            scheduler.step()

        model.load_state_dict(best_state_dict)
        model.eval()
        with torch.no_grad():
            test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
            outputs, attn_last, attn_rollout = model(
                test_data_batch_dev, return_attention=True, use_rollout=True
            )
            test_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            acc = metrics.accuracy_score(Y_test, test_preds)
            print("Test acc", acc)

        os.makedirs("./modelstf", exist_ok=True)
        torch.save(model.state_dict(), "./modelstf/" + str(kfold_index) + ".pt")
        result.append([kfold_index, acc])
        acc_all += acc
        time_train_end = time.time()
        time_train_list.append(time_train_end - time_train_start)

        if attn_last is not None and attn_rollout is not None:
            attn_last_np = attn_last.cpu().numpy()
            attn_rollout_np = attn_rollout.cpu().numpy()
            fold_attention_dir = os.path.join(attention_output_root, f"fold_{kfold_index:02d}")
            export_fold_attention_maps(
                attn_last_np,
                attn_rollout_np,
                Y_test,
                fold_attention_dir,
                node_labels=node_labels,
                asd_label=asd_label,
                hc_label=hc_label,
                top_k=top_k_nodes,
            )
            print(f"Saved attention maps to: {fold_attention_dir}")

            for label in (asd_label, hc_label):
                label_idx = np.where(Y_test == label)[0]
                if label_idx.size > 0:
                    global_last_sum[label] += attn_last_np[label_idx].sum(axis=0)
                    global_rollout_sum[label] += attn_rollout_np[label_idx].sum(axis=0)
                    global_count[label] += int(label_idx.size)

    temp = acc_all / outer_folds
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / len(result_final)
    print(result)

    global_attention_dir = os.path.join(attention_output_root, "global")
    os.makedirs(global_attention_dir, exist_ok=True)
    if global_count[asd_label] > 0 and global_count[hc_label] > 0:
        asd_last_global = global_last_sum[asd_label] / global_count[asd_label]
        hc_last_global = global_last_sum[hc_label] / global_count[hc_label]
        diff_last_global = asd_last_global - hc_last_global
        diff_last_bound = float(np.max(np.abs(diff_last_global)))

        save_attention_heatmap(
            asd_last_global,
            os.path.join(global_attention_dir, "asd_mean_last_layer_global.png"),
            "Global ASD Mean Attention - Last Layer Head Mean",
            cmap="viridis",
        )
        save_attention_heatmap(
            hc_last_global,
            os.path.join(global_attention_dir, "hc_mean_last_layer_global.png"),
            "Global HC Mean Attention - Last Layer Head Mean",
            cmap="viridis",
        )
        save_attention_heatmap(
            diff_last_global,
            os.path.join(global_attention_dir, "asd_minus_hc_last_layer_global.png"),
            "Global ASD - HC Attention Difference (Last Layer Head Mean)",
            cmap="coolwarm",
            center=0.0,
            vmin=-diff_last_bound,
            vmax=diff_last_bound,
        )

        asd_rollout_global = global_rollout_sum[asd_label] / global_count[asd_label]
        hc_rollout_global = global_rollout_sum[hc_label] / global_count[hc_label]
        diff_rollout_global = asd_rollout_global - hc_rollout_global
        diff_rollout_bound = float(np.max(np.abs(diff_rollout_global)))

        save_attention_heatmap(
            asd_rollout_global,
            os.path.join(global_attention_dir, "asd_mean_rollout_global.png"),
            "Global ASD Mean Attention - Rollout",
            cmap="viridis",
        )
        save_attention_heatmap(
            hc_rollout_global,
            os.path.join(global_attention_dir, "hc_mean_rollout_global.png"),
            "Global HC Mean Attention - Rollout",
            cmap="viridis",
        )
        save_attention_heatmap(
            diff_rollout_global,
            os.path.join(global_attention_dir, "asd_minus_hc_rollout_global.png"),
            "Global ASD - HC Attention Difference (Rollout)",
            cmap="coolwarm",
            center=0.0,
            vmin=-diff_rollout_bound,
            vmax=diff_rollout_bound,
        )

        save_top_nodes(
            asd_last_global,
            os.path.join(global_attention_dir, "top10_busy_nodes_asd_last_layer_global.csv"),
            node_labels,
            top_k=top_k_nodes,
            use_abs=False,
        )
        save_top_nodes(
            hc_last_global,
            os.path.join(global_attention_dir, "top10_busy_nodes_hc_last_layer_global.csv"),
            node_labels,
            top_k=top_k_nodes,
            use_abs=False,
        )
        save_top_nodes(
            diff_last_global,
            os.path.join(global_attention_dir, "top10_diff_nodes_abs_last_layer_global.csv"),
            node_labels,
            top_k=top_k_nodes,
            use_abs=True,
        )
        print(f"Saved global attention summary to: {global_attention_dir}")
    else:
        print("Skip global attention summary: one class has zero samples.")

print(result_final)
print(acc_final)
print(f"Ave Training Time: {np.mean(time_train_list)} seconds")
