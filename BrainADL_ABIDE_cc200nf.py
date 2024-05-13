import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random
from sklearn import metrics
import scipy.io as scio
from torch.optim import lr_scheduler
from NodeFormer import NodeFormer
from SGFormer import SGFormer
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
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

# 转换数据以适应 NodeFormer
# 构建邻接矩阵和特征矩阵
adj_matrices = []  # 存储每个样本的邻接矩阵
feature_matrices = []  # 存储每个样本的特征矩阵

def matrix_to_graph(matrix, label,threshold=0.2):
    num_nodes = matrix.shape[0]
    edge_index = []
    edge_weight = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # 只考虑上三角矩阵
            if abs(matrix[i, j]) > threshold:  # 使用阈值
                edge_index.append([i, j])
                edge_weight.append(matrix[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    x = torch.eye(num_nodes)  # 使用单位矩阵作为特征
    # 添加标签
    y = torch.tensor([int(label)], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    return data

graphs = [matrix_to_graph(matrix,label, threshold=0.2) for matrix, label in zip(X, Y)]

print('---------------------')
print('X Atlas1:', X.shape)
print('Y Atlas1:', Y.shape)
print('Graphs:', len(graphs))
print('---------------------')

'''
    超参数      
'''

epochs = 30
batch_size = 32
dropout = 0.5
lr = 0.005
decay = 0.01
result = []
acc_final = 0
result_final = []
model_dim = 32
in_channels = 200  # 根据你的数据调整
out_channels = 2   # 输出类别数
hidden_channels = 32  # 隐藏层通道数
num_heads = 2  # 多头注意力机制的头数
num_layers_former = 3
num_layers_gnn = 2


# input_dim = Nodes_Atlas1 * Nodes_Atlas1
input_dim = Nodes_Atlas1



'''
    模型训练
'''
from sklearn.model_selection import KFold
start_time = time.time()
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

        # X_trainval, X_test = X[trainval_index], X[test_index]
        # Y_trainval, Y_test = Y[trainval_index], Y[test_index]
        # 将数据分割为训练集和测试集
        train_graphs = [graphs[i] for i in trainval_index]
        test_graphs = [graphs[i] for i in test_index]

        # 使用 PyTorch Geometric 的 DataLoader 来批量处理图数据
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size)

        # for train_index, val_index in kf.split(X_trainval, Y_trainval):
        #     # 取消验证集
        #     X_train, X_val = X_trainval[:], X_trainval[:]
        #     Y_train, Y_val = Y_trainval[:], Y_trainval[:]

        # print('X_train', X_train.shape)
        # print('X_test', X_test.shape)
        # print('Y_train', Y_train.shape)
        # print('Y_test', Y_test.shape)
        print('train_graphs', len(train_graphs))
        print('test_graphs', len(test_graphs))
        print('train_loader', len(train_loader))
        print('test_loader', len(test_loader))

        # model = SimpleTransformerModel(input_dim, model_dim, num_classes=2)
        # model = NodeFormer(input_dim, model_dim, out_channels=2 )
        # model = NodeFormer(
        #     in_channels=input_dim,  # 输入特征维度
        #     hidden_channels=model_dim,  # 隐藏层特征维度
        #     out_channels=2,  # 输出特征维度，对应于分类任务的类别数
        #     num_layers=3,  # Transformer 层的数量
        #     num_heads=4,  # 注意力头的数量
        #     dropout=0.5,  # Dropout 比率
        #     # 其他参数保持默认
        # )
        model = SGFormer(
            in_channels=input_dim,  # 输入特征维度
            hidden_channels=model_dim,  # 隐藏层特征维度
            out_channels=2,  # 输出特征维度，对应于分类任务的类别数
            trans_num_layers=num_layers_former,  # Transformer 层的数量
            trans_num_heads=num_heads,  # 注意力头的数量
            trans_dropout=0.5,  # Dropout 比率
            gnn_num_layers=num_layers_gnn,  # GNN 层的数量
            # 其他参数保持默认
        )
        model.to(device)
        # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 每30个epochs将学习率乘以0.1
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            model.train()
            loss_train = 0
            for data in train_loader:  # 使用 DataLoader
                data = data.to(device)
                optimizer.zero_grad()
                # outputs = model(data.x, data.edge_index, data.edge_attr)
                outputs = model(data.x, data.edge_index,data.batch)
                loss = loss_fn(outputs, data.y)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            loss_train /= len(train_loader)
            if epoch % 10 == 0:
                print('epoch:', epoch, 'train loss:', loss_train)
            # scheduler.step()
            # val
            if epoch % 1 == 0:
                model.eval()
                correct = 0
                for data in test_loader:
                    data = data.to(device)
                    outputs = model(data.x, data.edge_index, data.batch)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == data.y).sum().item()
                acc = correct / len(test_loader.dataset)
                print('Test acc', acc)

        torch.save(model.state_dict(), './modelsnf/' + str(kfold_index) + '.pt')
        result.append([kfold_index, acc])
        acc_all += acc

    temp = acc_all / 10
    acc_final += temp
    result_final.append(temp)
    ACC = acc_final / 10
    print(result)
end_time = time.time()
print(result_final)
print(acc_final)
# 总训练时间
total_train_time = end_time - start_time
print(f"Total training time: {total_train_time:.2f} seconds")

