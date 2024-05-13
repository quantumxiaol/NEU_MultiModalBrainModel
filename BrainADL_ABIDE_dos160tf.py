import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random
from sklearn import metrics
import scipy.io as scio
from torch.optim import lr_scheduler

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


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes,feature_dim=2, num_heads=2, num_layers=1, dropout=0.25):
        super(SimpleTransformerModel, self).__init__()

        self.model_dim = model_dim
        # 输入层，将原始特征转换为模型维度
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.input_bn = nn.BatchNorm1d(model_dim)
        # 定义Transformer编码器层
        transformer_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        
        # 使用多个Transformer编码器层创建Transformer编码器
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_bn = nn.BatchNorm1d(model_dim)
        # 输出层
        # self.output_fc = nn.Linear(model_dim, num_classes)
        self.output_fc = nn.Linear(model_dim, feature_dim)

    def forward(self, x):
        # 将输入数据通过输入全连接层转换
        x = self.input_fc(x)  # [batch, seq_len, features]
        # 应用输入批量归一化
        x = x.permute(0, 2, 1) # 调整维度以匹配BatchNorm1d的输入要求
        x = self.input_bn(x)
        x = x.permute(0, 2, 1) # 恢复原始维度
        # 调整维度以符合Transformer的输入要求
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]

        # Transformer处理，每个输入元素（token）被转换
        x = self.transformer(x)  # 在这一步，模型内部处理Q (Query), K (Key), V (Value)

        # 对所有序列位置取平均，聚合序列信息
        x = x.mean(dim=0)  # [batch, features]
        x = self.output_bn(x)

        # 通过输出全连接层获得最终的分类结果
        x = self.output_fc(x)  # [batch, num_classes]

        return x


'''
    导入ABIDE数据
'''

print('loading ABIDE data...')
Data_atlas1 = scio.loadmat('./ABIDEdata/pcc_correlation_871_dos160_.mat')  # 更新为dos160数据路径
X = Data_atlas1['connectivity']
Y = np.loadtxt('./ABIDEdata/871_label_dos160.txt')  # 更新为dos160标签路径

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
dropout = 0.25
lr = 0.005
decay = 0.01
result = []
acc_final = 0
result_final = []
model_dim = batch_size
input_dim = Nodes_Atlas1 * Nodes_Atlas1
'''
    模型训练
'''
from sklearn.model_selection import KFold

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

        model = SimpleTransformerModel(input_dim, model_dim, num_classes=2)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)  # 每30个epochs将学习率乘以0.1
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

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
                train_data_batch = train_data_batch.reshape(train_data_batch.shape[0], -1)
                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_data_batch_dev = train_data_batch_dev.unsqueeze(1)  # 增加序列维度
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                optimizer.zero_grad()
                # outputs, rec = model(train_data_batch_dev)
                outputs = model(train_data_batch_dev)
                loss1 = loss_fn(outputs, train_label_batch_dev)
                loss = loss1
                loss_train += loss
                loss.backward()
                optimizer.step()

            loss_train /= num_batch
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())
            scheduler.step()
            # val
            if epoch % 1 == 0:
                model.eval()
                test_data_batch = X_test.reshape(X_test.shape[0], -1)  # 重塑为 (num_test_samples, 40000)
                test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                test_data_batch_dev = test_data_batch_dev.unsqueeze(1)  # 添加伪序列维度 (num_test_samples, 1, 40000)
                # test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                # outputs, _= model(test_data_batch_dev)
                outputs = model(test_data_batch_dev)
                _, indices = torch.max(outputs, dim=1)
                preds = indices.cpu()
                acc = metrics.accuracy_score(preds, Y_test)
                print('Test acc', acc)

        torch.save(model.state_dict(), './modelstfdos160/' + str(kfold_index) + '.pt')
        result.append([kfold_index, acc])
        acc_all += acc

    temp = acc_all / 10
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / 10
    print(result)
print(result_final)
print(acc_final)


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
    
# class SimpleTransformerModel(nn.Module):
#     def __init__(self, input_dim, model_dim, num_classes, num_heads=2, num_layers=1, dropout=0.1):
#         super(SimpleTransformerModel, self).__init__()

#         self.model_dim = model_dim
#         self.input_fc = nn.Linear(input_dim, model_dim)
#         transformer_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
#         self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
#         self.output_fc = nn.Linear(model_dim, num_classes)

#     def forward(self, x):
#         x = self.input_fc(x)
#         x = x.permute(1, 0, 2)  # Transformer期望的输入形状是 [seq_len, batch, features]
#         x = self.transformer(x)
#         x = x.mean(dim=0)  # 对所有序列位置取平均
#         x = self.output_fc(x)
#         return x