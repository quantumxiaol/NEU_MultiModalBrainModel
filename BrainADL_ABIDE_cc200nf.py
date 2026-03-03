import copy
import os
import random
import time

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from SGFormer import SGFormer

load_dotenv()
device = torch.device(os.getenv("DEVICE", "cpu"))
print(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def matrix_to_graph(matrix, label, threshold=0.2):
    num_nodes = matrix.shape[0]
    row, col = np.where(np.abs(matrix) > threshold)
    non_diag = row != col
    row = row[non_diag]
    col = col[non_diag]

    if row.size == 0:
        row = np.arange(num_nodes)
        col = np.arange(num_nodes)
        edge_weight = np.ones(num_nodes, dtype=np.float32)
    else:
        edge_weight = matrix[row, col].astype(np.float32)

    edge_index = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    x = torch.eye(num_nodes, dtype=torch.float32)
    y = torch.tensor([int(label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


def build_graphs(matrices, labels, threshold=0.2):
    return [matrix_to_graph(matrix, label, threshold=threshold) for matrix, label in zip(matrices, labels)]


def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(outputs, data.y)
            batch_graphs = data.num_graphs
            loss_sum += loss.item() * batch_graphs
            sample_count += batch_graphs
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    mean_loss = loss_sum / max(sample_count, 1)
    acc = metrics.accuracy_score(all_labels, all_preds) if sample_count > 0 else 0.0
    return mean_loss, acc


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
batch_size = 32
dropout = 0.5
lr = 1e-4
decay = 0.01
outer_folds = 10
val_ratio = 0.1
early_stop_patience = 10
early_stop_min_delta = 1e-4
graph_threshold = 0.2

result = []
acc_final = 0.0
result_final = []
model_dim = 32
num_heads = 2
num_layers_former = 3
num_layers_gnn = 2
input_dim = Nodes_Atlas1
time_train_list = []

start_time = time.time()
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
    graphs = build_graphs(X, Y, threshold=graph_threshold)

    acc_all = 0.0
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    for kfold_index, (trainval_index, test_index) in enumerate(kf.split(X, Y), start=1):
        time_train_start = time.time()
        print("kfold_index:", kfold_index)

        train_idx, val_idx = train_test_split(
            trainval_index,
            test_size=val_ratio,
            random_state=42 + kfold_index,
            stratify=Y[trainval_index],
        )

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_index]

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        print("train_graphs", len(train_graphs))
        print("val_graphs", len(val_graphs))
        print("test_graphs", len(test_graphs))
        print("train_loader", len(train_loader))
        print("val_loader", len(val_loader))
        print("test_loader", len(test_loader))

        model = SGFormer(
            in_channels=input_dim,
            hidden_channels=model_dim,
            out_channels=2,
            trans_num_layers=num_layers_former,
            trans_num_heads=num_heads,
            trans_dropout=dropout,
            gnn_num_layers=num_layers_gnn,
        )
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_sample_count = 0

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                outputs = model(data.x, data.edge_index, data.batch)
                loss = loss_fn(outputs, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_graphs = data.num_graphs
                train_loss_sum += loss.item() * batch_graphs
                train_sample_count += batch_graphs

            train_loss = train_loss_sum / max(train_sample_count, 1)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    "epoch:",
                    epoch,
                    "train loss:",
                    train_loss,
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
        test_loss, acc = evaluate(model, test_loader, loss_fn, device)
        print("Test loss", test_loss, "Test acc", acc)

        os.makedirs("./modelsnf", exist_ok=True)
        torch.save(model.state_dict(), "./modelsnf/" + str(kfold_index) + ".pt")
        result.append([kfold_index, acc])
        acc_all += acc

        time_train_end = time.time()
        time_train_list.append(time_train_end - time_train_start)

    temp = acc_all / outer_folds
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / len(result_final)
    print(result)

end_time = time.time()
print(result_final)
print(ACC)
print(f"Total training time: {end_time - start_time:.2f} seconds")
print(f"Ave fold training time: {np.mean(time_train_list):.2f} seconds")
