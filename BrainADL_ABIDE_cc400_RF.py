import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import scipy.io as scio
from torch.optim import lr_scheduler
import time
from sklearn.metrics import roc_auc_score, recall_score, f1_score
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
acc_final = 0

epochs = 30
batch_size = 64
dropout = 0.25
lr = 0.005
decay = 0.01
num_heads=2
num_layers=1
feature_dim=2
model_dim = 64
input_dim = Nodes_Atlas1 * Nodes_Atlas1
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

        print('X_train', X_train.shape)
        print('X_test', X_test.shape)
        print('Y_train', Y_train.shape)
        print('Y_test', Y_test.shape)

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)
        acc = metrics.accuracy_score(preds, Y_test)
        probs = model.predict_proba(X_test)[:, 1]  # 假设第二列是正类的概率
        auc_score = roc_auc_score(Y_test, probs)
        recall = recall_score(Y_test, preds, average='macro', zero_division=1)
        f1 = f1_score(Y_test, preds, average='macro', zero_division=1)
        
        print('Test acc', acc, 'Test recall', recall, 'Test f1', f1, 'Test auc', auc_score)
        
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc_score)

        result.append([kfold_index, acc])
        acc_all += acc
        auc_k.append([kfold_index, auc_score])
        recall_k.append([kfold_index, recall])
        f1_k.append([kfold_index, f1])
        time_train_end = time.time()

    temp = acc_all / 10
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / 10
    print(result)
    time_train_list.append(time_train_end - time_train_start)
print(result_final)
print(acc_final)
print(f"Ave Recall: {np.mean(recall_list)}")
print(f"Ave F1: {np.mean(f1_list)}")
print(f"Ave AUC: {np.mean(auc_list)}")
print(f"Ave Training Time: {np.mean(time_train_list)} seconds")
    
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