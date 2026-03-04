import os
import random
import time

import numpy as np
import scipy.io as scio
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC


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


def select_svm_model(x_train, y_train, x_val, y_val):
    candidates = [
        {"C": 0.5, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 2.0, "gamma": "scale"},
    ]

    best_model = None
    best_params = None
    best_val_score = -np.inf

    for params in candidates:
        model = SVC(kernel="rbf", probability=True, random_state=42, **params)
        model.fit(x_train, y_train)
        val_probs = model.predict_proba(x_val)[:, 1]
        if np.unique(y_val).size > 1:
            val_score = roc_auc_score(y_val, val_probs)
        else:
            val_preds = model.predict(x_val)
            val_score = accuracy_score(y_val, val_preds)

        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model
            best_params = params

    return best_model, best_params, best_val_score


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

    outer_folds = 10
    val_ratio = 0.1

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

            x_train_flat = x_train.reshape(x_train.shape[0], -1)
            x_val_flat = x_val.reshape(x_val.shape[0], -1)
            x_test_flat = x_test.reshape(x_test.shape[0], -1)

            model, best_params, best_val_score = select_svm_model(
                x_train_flat,
                y_train,
                x_val_flat,
                y_val,
            )

            test_preds = model.predict(x_test_flat)
            test_probs = model.predict_proba(x_test_flat)[:, 1]
            acc, recall, f1, auc_score = compute_fold_metrics(y_test, test_preds, test_probs)

            print(
                "Test acc",
                acc,
                "Test recall",
                recall,
                "Test f1",
                f1,
                "Test auc",
                auc_score,
                "best_val_score",
                best_val_score,
                "best_params",
                best_params,
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
