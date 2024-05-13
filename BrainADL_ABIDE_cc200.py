import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random
from sklearn import metrics
import scipy.io as scio
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        x = F.softmax(x, dim=-1)

        return x, None

'''
    导入ABIDE数据
'''

print('loading ABIDE data...')
Data_atlas1 = scio.loadmat('./ABIDEdata/pcc_correlation_871_cc200_.mat')
X = Data_atlas1['connectivity']
Y = np.loadtxt('./ABIDEdata/871_label_cc200.txt')

Nodes_Atlas1 = X.shape[-1]

where_are_nan = np.isnan(X)
where_are_inf = np.isinf(X)

where_are_nan = np.isnan(X)
where_are_inf = np.isinf(X)
for bb in range(0, N_subjects):
    for i in range(0, Nodes_Atlas1):
        for j in range(0, Nodes_Atlas1):
            if where_are_nan[bb][i][j]:
                X[bb][i][j] = 0
            if where_are_inf[bb][i][j]:
                X[bb][i][j] = 1

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
result = []
acc_final = 0
result_final = []

'''
    模型训练
'''
from sklearn.model_selection import KFold
time_train_list = []
time_data_load_end = time.time()
for ind in range(1):
    setup_seed(ind)

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
    kf = KFold(n_splits=10, shuffle=True)
    kfold_index = 0
    for trainval_index, test_index in kf.split(X, Y):
        kfold_index += 1
        time_train_start = time.time()
        print('kfold_index:', kfold_index)

        X_trainval, X_test = X[trainval_index], X[test_index]
        Y_trainval, Y_test = Y[trainval_index], Y[test_index]

        for train_index, val_index in kf.split(X_trainval, Y_trainval):
            # 取消验证集
            X_train, X_val = X_trainval[:], X_trainval[:]
            Y_train, Y_val = Y_trainval[:], Y_trainval[:]

        print('X_train', X_train.shape)
        print('X_test', X_test.shape)
        print('Y_train', Y_train.shape)
        print('Y_test', Y_test.shape)

        model = Model(dropout=dropout, num_class=2)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            model.train()

            idx_batch = np.random.permutation(int(X_train.shape[0]))
            num_batch = X_train.shape[0] // int(batch_size)

            loss_train = 0
            for bn in range(num_batch):
                if bn == num_batch - 1:
                    batch = idx_batch[bn * int(batch_size):]
                else:
                    batch = idx_batch[bn * int(batch_size): (bn + 1) * int(batch_size)]
                train_data_batch = X_train[batch]
                train_label_batch = Y_train[batch]

                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                optimizer.zero_grad()
                outputs, rec = model(train_data_batch_dev)
                loss1 = loss_fn(outputs, train_label_batch_dev)
                loss = loss1
                loss_train += loss
                loss.backward()
                optimizer.step()

            loss_train /= num_batch
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())

            # val
            if epoch % 1 == 0:
                model.eval()
                test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                outputs, _= model(test_data_batch_dev)
                _, indices = torch.max(outputs, dim=1)
                preds = indices.cpu()
                acc = metrics.accuracy_score(preds, Y_test)
                print('Test acc', acc)

        torch.save(model.state_dict(), './models/' + str(kfold_index) + '.pt')
        result.append([kfold_index, acc])
        acc_all += acc
        time_train_end = time.time()
        time_train_list.append(time_train_end - time_train_start)
    temp = acc_all / 10
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / 10
    print(result)
print(result_final)
print(acc_final)
print(f"Ave Training Time: {np.mean(time_train_list)} seconds")