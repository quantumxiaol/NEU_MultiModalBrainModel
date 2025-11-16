import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random
from sklearn import metrics
import scipy.io as scio
from torch.optim import lr_scheduler
import time
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score
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

# https://github.com/Wayfear/BrainNetworkTransformer/blob/main/source/models/brainnetcnn.py

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, node_sz):
        super().__init__() 
        self.in_planes = 1
        self.d = node_sz

        self.e2econv1 = E2EBlock(1, 8, self.d, bias=True)
        self.e2econv2 = E2EBlock(8, 16, self.d, bias=True)
        self.E2N = torch.nn.Conv2d(16, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 64, (self.d, 1))
        self.dense1 = torch.nn.Linear(64, 32)
        self.dense2 = torch.nn.Linear(32, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self, x):
        # x = x.unsqueeze(dim=1)  # 增加通道维度
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        return out


'''
    导入ABIDE数据
'''

print('loading ABIDE data...')
Data_atlas1 = scio.loadmat('./ABIDEdata/pcc_correlation_871_cc400_.mat')  # 更新为cc400数据路径
X = Data_atlas1['connectivity']
Y = np.loadtxt('./ABIDEdata/871_label_cc400.txt')  # 更新为cc400标签路径

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
result = []
acc_k=[]
f1_k=[]
recall_k=[]
auc_k=[]
acc_final = 0
result_final = []
recall_list = []
f1_list = []
auc_list = []


epochs = 30
batch_size = 2
dropout = 0.25
lr = 0.005
decay = 0.001
acc_final = 0
model_dim = 64
input_dim = Nodes_Atlas1 * Nodes_Atlas1
node_sz = Nodes_Atlas1

'''
    模型训练
'''
from sklearn.model_selection import KFold
time_train_list = []
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

        X_train_reshaped = X_train.reshape(-1, 1, node_sz, node_sz)
        X_test_reshaped = X_test.reshape(-1, 1, node_sz, node_sz)

        print('X_train', X_train.shape)
        print('X_test', X_test.shape)
        print('X_train_reshaped', X_train_reshaped.shape)
        print('X_test_reshaped', X_test_reshaped.shape)
        print('Y_train', Y_train.shape)
        print('Y_test', Y_test.shape)

        # model = SimpleTransformerModel(input_dim, model_dim, num_classes=2)
        model = BrainNetCNN(node_sz=node_sz)
        model.to(device)
        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
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
                # train_data_batch = X_train[batch]
                # train_label_batch = Y_train[batch]
                # train_data_batch = train_data_batch.reshape(train_data_batch.shape[0], -1)
                # train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                # train_data_batch_dev = train_data_batch_dev.unsqueeze(1)  # 增加序列维度
                # train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                train_data_batch = X_train_reshaped[batch]
                train_label_batch = Y_train[batch]
        
                # 无需重新塑形为向量再变回2D
                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)
                # print(train_data_batch_dev.shape)
                # print(train_label_batch_dev.shape)


                optimizer.zero_grad()
                # outputs, rec = model(train_data_batch_dev)
                outputs = model(train_data_batch_dev)
                loss1 = loss_fn(outputs, train_label_batch_dev)
                loss = loss1
                loss_train += loss
                loss.backward()
                optimizer.step()

            loss_train /= num_batch
            if epoch % 1 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item())
            scheduler.step()
            # val
            if epoch % 5 == 0:
                model.eval()
                # test_data_batch = X_test.reshape(X_test.shape[0], -1)  # 重塑为 (num_test_samples, 40000)
                # test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                # test_data_batch_dev = test_data_batch_dev.unsqueeze(1)  # 添加伪序列维度 (num_test_samples, 1, 40000)
                # test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                # outputs, _= model(test_data_batch_dev)
                test_data_batch_dev = torch.from_numpy(X_test_reshaped).float().to(device)



                outputs = model(test_data_batch_dev)
                _, indices = torch.max(outputs, dim=1)
                preds = indices.cpu()
                acc = metrics.accuracy_score(preds, Y_test)
                # recall = recall_score(preds, Y_test, average='macro')  # 可以根据需要选择average参数
                # f1 = f1_score(preds, Y_test, average='macro')  # 可以根据需要选择average参数
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 假设第二列是正类的概率
                probs_list = probs.detach().cpu().numpy()
                # y_true_list = Y1_test.cpu().numpy()

                # 检查y_true中是否有多于一个类别
                if len(np.unique(Y_test)) > 1:
                    auc_score = roc_auc_score(Y_test, probs_list)
                    auc_list.append(auc_score)
                else:
                    # 处理只有一个类别的情况
                    # print(f"Only one class present in fold {kfold_index}. AUC not defined.")
                    # 选择性地添加默认值，例如
                    auc_list.append(0.5)  # 或者其他您认为合适的默认值
                # auc = roc_auc_score(preds, Y1_test)
                recall = recall_score(preds, Y_test, average='macro', zero_division=1)
                f1 = f1_score(preds, Y_test, average='macro', zero_division=1)
                print('Test acc', acc, 'Test recall', recall, 'Test f1', f1, 'Test auc', auc_score)
                recall_list.append(recall)
                f1_list.append(f1)
        os.makedirs('./modelsBrainNetcc400', exist_ok=True)

        torch.save(model.state_dict(), './modelsBrainNetcc400/' + str(kfold_index) + '.pt')
        result.append([kfold_index, acc])
        auc_k.append([kfold_index, auc_score])
        recall_k.append([kfold_index, recall])
        f1_k.append([kfold_index, f1])
        acc_all += acc
        time_train_end = time.time()
    temp = acc_all / 10
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / 10
    print("acc",result)
    print("recall",recall_k)
    print("f1",f1_k)
    print("AUC",auc_k)
    
    time_train_list.append(time_train_end - time_train_start)
print(result_final)
print(acc_final)
# print("recall",recall_list)
print(f"Ave Recall: {np.mean(recall_list)}")
# print("F1",f1_list)
print(f"Ave F1: {np.mean(f1_list)}")
# print("AUC",auc_list)
print(f"Ave AUC: {np.mean(auc_list)}")
print(f"Ave Training Time: {np.mean(time_train_list)} seconds")


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