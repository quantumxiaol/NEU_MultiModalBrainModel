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
from sklearn.model_selection import KFold, train_test_split
from torch.optim import lr_scheduler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_and_preprocess_data(atlas_path, label_path, hc_label=0):
    data_atlas = scio.loadmat(atlas_path)
    x = data_atlas["connectivity"]
    y = np.loadtxt(label_path).astype(np.int64)

    finite_mask = np.isfinite(x)
    normal_subject_mask = y == hc_label
    reference_x = x[normal_subject_mask] if np.any(normal_subject_mask) else x
    reference_finite_mask = np.isfinite(reference_x)
    mean_connectivity = np.nanmean(np.where(reference_finite_mask, reference_x, np.nan), axis=0)
    mean_connectivity = np.nan_to_num(mean_connectivity, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.where(finite_mask, x, mean_connectivity[np.newaxis, :, :]).astype(np.float32)

    return x, y


def compute_fold_metrics(y_true, preds, probs):
    acc = accuracy_score(y_true, preds)
    recall = recall_score(y_true, preds, average="macro", zero_division=1)
    f1 = f1_score(y_true, preds, average="macro", zero_division=1)
    if np.unique(y_true).size > 1:
        auc_score = roc_auc_score(y_true, probs)
    else:
        auc_score = 0.5
    return acc, recall, f1, auc_score


class E2EBlock(nn.Module):
    def __init__(self, in_planes, out_planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = nn.Conv2d(in_planes, out_planes, (1, self.d), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, out_planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, dim=3) + torch.cat([b] * self.d, dim=2)


class BrainNetCNN(nn.Module):
    def __init__(self, node_sz, dropout=0.25):
        super().__init__()
        self.e2econv1 = E2EBlock(1, 8, node_sz, bias=True)
        self.e2econv2 = E2EBlock(8, 16, node_sz, bias=True)
        self.e2n = nn.Conv2d(16, 32, (1, node_sz))
        self.n2g = nn.Conv2d(32, 64, (node_sz, 1))

        self.drop = nn.Dropout(dropout)
        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, 2)

    def forward(self, x):
        # x: [batch, num_nodes, num_nodes]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.e2n(out), negative_slope=0.33)
        out = self.drop(F.leaky_relu(self.n2g(out), negative_slope=0.33))

        out = out.view(out.size(0), -1)
        out = self.drop(F.leaky_relu(self.dense1(out), negative_slope=0.33))
        out = self.drop(F.leaky_relu(self.dense2(out), negative_slope=0.33))
        out = self.dense3(out)
        return out


def main():
    load_dotenv()
    device = torch.device(os.getenv("DEVICE", "cpu"))
    print(device)

    setup_seed(123)

    data_load_start = time.time()
    print("loading ABIDE data...")
    x, y = load_and_preprocess_data(
        atlas_path="./ABIDEdata/pcc_correlation_871_cc400_.mat",
        label_path="./ABIDEdata/871_label_cc400.txt",
    )
    data_load_end = time.time()

    print("---------------------")
    print("X Atlas1:", x.shape)
    print("Y Atlas1:", y.shape)
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

    os.makedirs("./modelsBrainNetcc400", exist_ok=True)

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
        kf = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

        for kfold_index, (trainval_index, test_index) in enumerate(kf.split(x, y), start=1):
            time_train_start = time.time()
            print("kfold_index:", kfold_index)

            x_trainval, x_test = x[trainval_index], x[test_index]
            y_trainval, y_test = y[trainval_index], y[test_index]

            x_train, x_val, y_train, y_val = train_test_split(
                x_trainval,
                y_trainval,
                test_size=val_ratio,
                random_state=42 + kfold_index,
                stratify=y_trainval,
            )

            print("X_train", x_train.shape)
            print("X_val", x_val.shape)
            print("X_test", x_test.shape)
            print("Y_train", y_train.shape)
            print("Y_val", y_val.shape)
            print("Y_test", y_test.shape)

            model = BrainNetCNN(node_sz=x.shape[-1], dropout=dropout).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            loss_fn = nn.CrossEntropyLoss()

            best_val_loss = float("inf")
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            for epoch in range(1, epochs + 1):
                model.train()
                idx_batch = np.random.permutation(x_train.shape[0])
                loss_train_list = []

                for start in range(0, x_train.shape[0], batch_size):
                    batch = idx_batch[start : start + batch_size]
                    train_data_batch = torch.from_numpy(x_train[batch]).float().to(device)
                    train_label_batch = torch.from_numpy(y_train[batch]).long().to(device)

                    optimizer.zero_grad()
                    outputs = model(train_data_batch)
                    loss = loss_fn(outputs, train_label_batch)
                    loss_train_list.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_train = float(np.mean(loss_train_list)) if loss_train_list else 0.0

                model.eval()
                with torch.no_grad():
                    val_data_batch = torch.from_numpy(x_val).float().to(device)
                    val_label_batch = torch.from_numpy(y_val).long().to(device)
                    val_outputs = model(val_data_batch)
                    val_loss = loss_fn(val_outputs, val_label_batch).item()
                    val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                    val_acc = accuracy_score(y_val, val_preds)

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
                test_data_batch = torch.from_numpy(x_test).float().to(device)
                outputs = model(test_data_batch)
                test_probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                test_preds = torch.argmax(outputs, dim=1).cpu().numpy()

            acc, recall, f1, auc_score = compute_fold_metrics(y_test, test_preds, test_probs)
            print("Test acc", acc, "Test recall", recall, "Test f1", f1, "Test auc", auc_score)

            torch.save(model.state_dict(), f"./modelsBrainNetcc400/{kfold_index}.pt")

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

    print("acc", result)
    print("recall", recall_k)
    print("f1", f1_k)
    print("AUC", auc_k)
    print(result_final)
    print(acc_final)
    print(f"Ave Recall: {np.mean(recall_list)}")
    print(f"Ave F1: {np.mean(f1_list)}")
    print(f"Ave AUC: {np.mean(auc_list)}")
    print(f"Data Loading Time: {data_load_end - data_load_start} seconds")
    print(f"Ave Training Time: {np.mean(time_train_list)} seconds")


if __name__ == "__main__":
    main()
