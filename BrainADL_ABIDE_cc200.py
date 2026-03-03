import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import copy
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
import scipy.io as scio
import time
import os
from dotenv import load_dotenv
load_dotenv()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(os.getenv("DEVICE","cpu"))
print(device)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

N_subjects = 871

class E2E(nn.Module):

    def __init__(self, in_channel, out_channel, input_shape):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.d = input_shape[0]
        self.conv1xd = nn.Conv2d(in_channel,out_channel,(self.d,1))
        self.convdx1 = nn.Conv2d(in_channel,out_channel,(1,self.d))
        self.node = 200

    def forward(self, A):
        A = A.view(-1, self.in_channel,self.node,self.node)
        a = self.conv1xd(A)
        b = self.convdx1(A)

        concat1 = torch.cat([a]*self.d,2) #d个a []括号内 竖着拼 d*d
        concat2 = torch.cat([b]*self.d,3) #横着拼 d*d

        return concat1+concat2

class Model(nn.Module):
    def __init__(self, dropout=0.5, num_class=1, nodes=200):
        super().__init__()

        self.e2e = nn.Sequential(
            E2E(1, 8, (nodes, nodes)), #1 32   32 56    56 32    32 2
            nn.LeakyReLU(0.33),
            E2E(8, 8, (nodes, nodes)),
            nn.LeakyReLU(0.33),
        )

        self.e2n = nn.Sequential(
            nn.Conv2d(8, 48,(1, nodes)), # 56 0.602
            nn.LeakyReLU(0.33)
        )

        self.n2g = nn.Sequential(
            nn.Conv2d(48, nodes,(nodes, 1)),
            nn.LeakyReLU(0.33)
        )

        self.linear = nn.Sequential(
            nn.Linear(nodes, 64),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(64, 10),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.33),
            nn.Linear(10, num_class)
        )

        for layer in self.linear:
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.e2e(x)
        x = self.e2n(x)
        x = self.n2g(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, None

'''
    导入ABIDE数据
'''

print('loading ABIDE data...')
Data_atlas1 = scio.loadmat('./ABIDEdata/pcc_correlation_871_cc200_.mat')
X = Data_atlas1['connectivity']
Y = np.loadtxt('./ABIDEdata/871_label_cc200.txt').astype(np.int64)

Nodes_Atlas1 = X.shape[-1]

finite_mask = np.isfinite(X)
normal_subject_mask = (Y == 0)
reference_X = X[normal_subject_mask] if np.any(normal_subject_mask) else X
reference_finite_mask = np.isfinite(reference_X)
mean_connectivity = np.nanmean(np.where(reference_finite_mask, reference_X, np.nan), axis=0)
mean_connectivity = np.nan_to_num(mean_connectivity, nan=0.0, posinf=0.0, neginf=0.0)
X = np.where(finite_mask, X, mean_connectivity[np.newaxis, :, :])

print('---------------------')
print('X Atlas1:', X.shape)
print('Y Atlas1:', Y.shape)
print('---------------------')

'''
    超参数      
'''

epochs = 30
batch_size = 64
dropout = 0.5
lr = 0.005
decay = 0.01
outer_folds = 10
val_ratio = 0.1
early_stop_patience = 10
early_stop_min_delta = 1e-4
result = []
acc_final = 0
result_final = []

'''
    模型训练
'''
time_train_list = []
for run_idx in range(1):
    setup_seed(run_idx)

    nodes_number = Nodes_Atlas1
    nums = np.ones(Nodes_Atlas1)
    nums[:Nodes_Atlas1 - nodes_number] = 0
    np.random.seed(1)
    np.random.shuffle(nums)
    Mask = nums.reshape(nums.shape[0], 1) * nums
    Masked_X = X * Mask
    J = nodes_number
    for i in range(0, J):
        ind = i
        if nums[ind] == 0:
            for j in range(J, Nodes_Atlas1):
                if nums[j] == 1:
                    Masked_X[:, [ind, j], :] = Masked_X[:, [j, ind], :]
                    Masked_X[:, :, [ind, j]] = Masked_X[:, :, [j, ind]]
                    J = j + 1
                    break

    Masked_X = Masked_X[:, :nodes_number, :nodes_number]
    X = Masked_X

    acc_all = 0
    kf = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    for kfold_index, (trainval_index, test_index) in enumerate(kf.split(X, Y), start=1):
        time_train_start = time.time()
        print('kfold_index:', kfold_index)

        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_trainval,
            Y_trainval,
            test_size=val_ratio,
            random_state=42 + kfold_index,
            stratify=Y_trainval,
        )

        print('X_train', X_train.shape)
        print('X_val', X_val.shape)
        print('X_test', X_test.shape)
        print('Y_train', Y_train.shape)
        print('Y_val', Y_val.shape)
        print('Y_test', Y_test.shape)

        model = Model(dropout=dropout, num_class=2)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        loss_fn = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()

            idx_batch = np.random.permutation(int(X_train.shape[0]))
            loss_train_list = []
            for start in range(0, X_train.shape[0], int(batch_size)):
                batch = idx_batch[start:start + int(batch_size)]
                train_data_batch = X_train[batch]
                train_label_batch = Y_train[batch]

                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                optimizer.zero_grad()
                outputs, rec = model(train_data_batch_dev)
                loss1 = loss_fn(outputs, train_label_batch_dev)
                loss = loss1
                loss_train_list.append(loss.item())
                loss.backward()
                optimizer.step()

            loss_train = float(np.mean(loss_train_list)) if loss_train_list else 0.0

            model.eval()
            with torch.no_grad():
                val_data_batch_dev = torch.from_numpy(X_val).float().to(device)
                val_label_batch_dev = torch.from_numpy(Y_val).long().to(device)
                val_outputs, _ = model(val_data_batch_dev)
                val_loss = loss_fn(val_outputs, val_label_batch_dev).item()
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_acc = metrics.accuracy_score(Y_val, val_preds)

            if epoch % 5 == 0 or epoch == 1:
                print('epoch:', epoch, 'train loss:', loss_train, 'val loss:', val_loss, 'val acc:', val_acc)

            if val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f'Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}')
                    break

        model.load_state_dict(best_state_dict)
        model.eval()
        with torch.no_grad():
            test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
            test_outputs, _ = model(test_data_batch_dev)
            test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
            acc = metrics.accuracy_score(Y_test, test_preds)
            print('Test acc', acc)

        os.makedirs('./models', exist_ok=True)
        torch.save(model.state_dict(), './models/' + str(kfold_index) + '.pt')
        result.append([kfold_index, acc])
        acc_all += acc
        time_train_end = time.time()
        time_train_list.append(time_train_end - time_train_start)
    temp = acc_all / outer_folds
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / len(result_final)
    print(result)
print(result_final)
print(ACC)
print(f"Ave Training Time: {np.mean(time_train_list)} seconds")
