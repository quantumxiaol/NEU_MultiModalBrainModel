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
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score,roc_auc_score

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
    def __init__(self, input_dim, model_dim, num_classes,feature_dim, num_heads=4, num_layers=2, dropout=0.25):
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

# class MultiModalBrainModel(nn.Module):
#     def __init__(self, models, fusion_dim,feature_dim, num_classes):
#         super(MultiModalBrainModel, self).__init__()
#         self.models = nn.ModuleList(models)
#         self.batch_norm = nn.BatchNorm1d(len(models) * feature_dim)
#         self.fusion_layer = nn.Linear(len(models) * feature_dim, fusion_dim)
#         self.classifier = nn.Linear(fusion_dim, num_classes)

#     def forward(self, x_list):
#         model_outputs = [model(x) for model, x in zip(self.models, x_list)]
#         # print("Each model output shape:", [output.shape for output in model_outputs])
#         fused = torch.cat(model_outputs, dim=1)
#         fused = self.batch_norm(fused)
#         fused = F.relu(self.fusion_layer(fused))
#         output = self.classifier(fused)
#         return output

# class MultiModalBrainModel(nn.Module):
#     def __init__(self, models, fusion_dim, feature_dim, num_classes):
#         super(MultiModalBrainModel, self).__init__()
#         self.models = nn.ModuleList(models)

#         # 相似性编码器
#         self.similarity_encoder = nn.Sequential(
#             nn.BatchNorm1d(len(models) * feature_dim),
#             nn.Linear(len(models) * feature_dim, fusion_dim),
#             nn.ReLU()
#         )

#         # 差异性编码器
#         self.difference_encoder = nn.Sequential(
#             nn.BatchNorm1d(len(models) * feature_dim),
#             nn.Linear(len(models) * feature_dim, fusion_dim),
#             nn.ReLU()
#         )

#         # 将相似性信息用于最终分类
#         self.classifier = nn.Linear(fusion_dim, num_classes)

#     def forward(self, x_list):
#         model_outputs = [model(x) for model, x in zip(self.models, x_list)]

#         # 将不同模型的输出合并
#         fused = torch.cat(model_outputs, dim=1)

#         # 计算相似性和差异性编码
#         similarity = self.similarity_encoder(fused)
#         difference = self.difference_encoder(fused)

#         # 可以在此处设计一种策略来综合考虑相似性和差异性
#         # 例如，可以将它们相加、相减或者采用其他结合方式
#         # 这里我们只使用相似性编码进行分类
#         # 使用相似性编码的输出进行分类
#         class_output = self.classifier(similarity)
#         # class_output = self.classifier(difference)

#         # 返回分类输出、相似性编码和差异性编码的输出
#         return class_output, similarity, difference

class MultiModalBrainModel(nn.Module):
    def __init__(self, models, fusion_dim, feature_dim, num_classes, model_dim, num_heads=2, num_layers=2, dropout=0.5):
        super(MultiModalBrainModel, self).__init__()
        self.models = nn.ModuleList(models)

        # Transformer编码器层的配置
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=num_layers
        )

        # 相似性编码器
        self.similarity_encoder = nn.Sequential(
            nn.BatchNorm1d(len(models) * feature_dim),
            nn.Linear(len(models) * feature_dim, model_dim),
            self.transformer_encoder,
            nn.ReLU(),
            nn.Linear(model_dim, fusion_dim)
        )

        # 差异性编码器
        self.difference_encoder = nn.Sequential(
            nn.BatchNorm1d(len(models) * feature_dim),
            nn.Linear(len(models) * feature_dim, model_dim),
            self.transformer_encoder,
            nn.ReLU(),
            nn.Linear(model_dim, fusion_dim)
        )

        # 将相似性信息用于最终分类
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_list):
        model_outputs = [model(x) for model, x in zip(self.models, x_list)]
        # model_outputs[1] *= 1.005
        fused = torch.cat(model_outputs, dim=1)

        similarity = self.similarity_encoder(fused)
        difference = self.difference_encoder(fused)

        # 在这里可以设计综合考虑相似性和差异性的策略
        # 例如，可以使用相似性编码进行分类
        class_output = self.classifier(similarity)
        # class_output = self.classifier(difference)

        return class_output, similarity, difference

def similarity_loss(output_list, alpha=0.5):
    """
    计算输出列表中所有元素的平均余弦相似性损失。
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # 在第二维（num_classes维度）上计算相似性
    loss = 0.0
    n = len(output_list)
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            # 计算两个输出之间的余弦相似度
            # 这将返回一个形状为 [batch] 的向量，每个样本一个相似度分数
            sim = cos(output_list[i], output_list[j])

            # 计算所有样本的平均相似性损失
            loss += (1 - sim).mean()
            count += 1

    return alpha * (loss / max(count, 1))

def contrastive_loss(output1, output2, margin=10):
    # 计算欧式距离
    distance = F.pairwise_distance(output1, output2)
    # print('distance:', distance)
    # 对于相同类别的样本（Y=1），减小距离；对于不同类别的样本（Y=0），增加距离
    loss = torch.mean(torch.clamp(margin - distance, min=0.0).pow(2))
    return loss

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2 , weight_temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.weight_temperature = weight_temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, anchor, positive, negatives):
        """
        anchor: Tensor representing anchor embeddings.
        positive: Tensor representing positive sample embeddings.
        negatives: Tensor representing negative sample embeddings.
        """
        anchor = anchor.unsqueeze(1)
        positive = positive.unsqueeze(1)

        positives_similarity = self.cosine_similarity(anchor, positive) / self.temperature
        negatives_similarity = self.cosine_similarity(anchor, negatives) / self.temperature

        negatives_similarity = negatives_similarity.logsumexp(dim=1)
        loss = -positives_similarity + negatives_similarity

        return loss.mean()

'''
    导入ABIDE数据
'''
time_data_load_start = time.time()
print('loading ABIDE data aal...')
Data_atlas1 = scio.loadmat('./ABIDEdata/pcc_correlation_871_aal_.mat')  # 更新为aal数据路径
X1 = Data_atlas1['connectivity']
Y1 = np.loadtxt('./ABIDEdata/871_label_aal.txt')  # 更新为aal标签路径
Nodes_Atlas1 = X1.shape[-1]
where_are_nan = np.isnan(X1)
where_are_inf = np.isinf(X1)
where_are_nan = np.isnan(X1)
where_are_inf = np.isinf(X1)
for bb in range(0, N_subjects):
    for i in range(0, Nodes_Atlas1):
        for j in range(0, Nodes_Atlas1):
            if where_are_nan[bb][i][j]:
                X1[bb][i][j] = 0
            if where_are_inf[bb][i][j]:
                X1[bb][i][j] = 1

print('---------------------')
print('X Atlas1:', X1.shape)
print('Y Atlas1:', Y1.shape)
print('---------------------')


# print('loading ABIDE data cc200...')
# Data_atlas2 = scio.loadmat('./ABIDEdata/pcc_correlation_871_cc200_.mat')  # 更新为cc200数据路径
# X2 = Data_atlas2['connectivity']
# Y2 = np.loadtxt('./ABIDEdata/871_label_cc200.txt')  # 更新为cc200标签路径
# Nodes_Atlas2 = X2.shape[-1]
# where_are_nan = np.isnan(X2)
# where_are_inf = np.isinf(X2)
# where_are_nan = np.isnan(X2)
# where_are_inf = np.isinf(X2)
# for bb in range(0, N_subjects):
#     for i in range(0, Nodes_Atlas2):
#         for j in range(0, Nodes_Atlas2):
#             if where_are_nan[bb][i][j]:
#                 X2[bb][i][j] = 0
#             if where_are_inf[bb][i][j]:
#                 X2[bb][i][j] = 1

# print('---------------------')
# print('X Atlas2:', X2.shape)
# print('Y Atlas2:', Y2.shape)
# print('---------------------')

print('loading ABIDE data cc400...')
Data_atlas3 = scio.loadmat('./ABIDEdata/pcc_correlation_871_cc400_.mat')  # 更新为cc400数据路径
X3 = Data_atlas3['connectivity']
Y3 = np.loadtxt('./ABIDEdata/871_label_cc400.txt')  # 更新为cc400标签路径
Nodes_Atlas3 = X3.shape[-1]
where_are_nan = np.isnan(X3)
where_are_inf = np.isinf(X3)
where_are_nan = np.isnan(X3)
where_are_inf = np.isinf(X3)
for bb in range(0, N_subjects):
    for i in range(0, Nodes_Atlas3):
        for j in range(0, Nodes_Atlas3):
            if where_are_nan[bb][i][j]:
                X3[bb][i][j] = 0
            if where_are_inf[bb][i][j]:
                X3[bb][i][j] = 1

print('---------------------')
print('X Atlas3:', X3.shape)   
print('Y Atlas3:', Y3.shape)
print('---------------------')

# print('loading ABIDE data dos160...')
# Data_atlas4 = scio.loadmat('./ABIDEdata/pcc_correlation_871_dos160_.mat')  # 更新为dos数据路径
# X4 = Data_atlas4['connectivity']
# Y4 = np.loadtxt('./ABIDEdata/871_label_dos160.txt')  # 更新为dos标签路径
# Nodes_Atlas4 = X4.shape[-1]
# where_are_nan = np.isnan(X4)
# where_are_inf = np.isinf(X4)
# where_are_nan = np.isnan(X4)
# where_are_inf = np.isinf(X4)
# for bb in range(0, N_subjects):
#     for i in range(0, Nodes_Atlas4):
#         for j in range(0, Nodes_Atlas4):
#             if where_are_nan[bb][i][j]:
#                 X4[bb][i][j] = 0
#             if where_are_inf[bb][i][j]:
#                 X4[bb][i][j] = 1

# print('---------------------')
# print('X Atlas4:', X4.shape)
# print('Y Atlas4:', Y4.shape)
# print('---------------------')

print('loading ABIDE data ez...')
Data_atlas5 = scio.loadmat('./ABIDEdata/pcc_correlation_871_ez_.mat')  # 更新为ez数据路径
X5 = Data_atlas5['connectivity']
Y5 = np.loadtxt('./ABIDEdata/871_label_ez.txt')  # 更新为ez标签路径
Nodes_Atlas5 = X5.shape[-1]
where_are_nan = np.isnan(X5)
where_are_inf = np.isinf(X5)
where_are_nan = np.isnan(X5)
where_are_inf = np.isinf(X5)
for bb in range(0, N_subjects):
    for i in range(0, Nodes_Atlas5):
        for j in range(0, Nodes_Atlas5):
            if where_are_nan[bb][i][j]:
                X5[bb][i][j] = 0
            if where_are_inf[bb][i][j]:
                X5[bb][i][j] = 1

print('---------------------')
print('X Atlas5:', X5.shape)
print('Y Atlas5:', Y5.shape)
print('---------------------')

# print('loading ABIDE data ho...')
# Data_atlas6 = scio.loadmat('./ABIDEdata/pcc_correlation_871_ho_.mat')  # 更新为ho数据路径
# X6 = Data_atlas6['connectivity']
# Y6 = np.loadtxt('./ABIDEdata/871_label_ho.txt')  # 更新为ho标签路径
# Nodes_Atlas6 = X6.shape[-1]
# where_are_nan = np.isnan(X6)
# where_are_inf = np.isinf(X6)
# where_are_nan = np.isnan(X6)
# where_are_inf = np.isinf(X6)
# for bb in range(0, N_subjects):
#     for i in range(0, Nodes_Atlas6):
#         for j in range(0, Nodes_Atlas6):
#             if where_are_nan[bb][i][j]:
#                 X6[bb][i][j] = 0
#             if where_are_inf[bb][i][j]:
#                 X6[bb][i][j] = 1

# print('---------------------')
# print('X Atlas6:', X6.shape)
# print('Y Atlas6:', Y6.shape)
# print('---------------------')

# print('loading ABIDE data tt...')
# Data_atlas7 = scio.loadmat('./ABIDEdata/pcc_correlation_871_tt_.mat')  # 更新为tt数据路径
# X7 = Data_atlas7['connectivity']
# Y7 = np.loadtxt('./ABIDEdata/871_label_tt.txt')  # 更新为tt标签路径
# Nodes_Atlas7 = X7.shape[-1]
# where_are_nan = np.isnan(X7)
# where_are_inf = np.isinf(X7)
# where_are_nan = np.isnan(X7)
# where_are_inf = np.isinf(X7)
# for bb in range(0, N_subjects):
#     for i in range(0, Nodes_Atlas7):
#         for j in range(0, Nodes_Atlas7):
#             if where_are_nan[bb][i][j]:
#                 X7[bb][i][j] = 0
#             if where_are_inf[bb][i][j]:
#                 X7[bb][i][j] = 1

# print('---------------------')
# print('X Atlas7:', X7.shape)
# print('Y Atlas7:', Y7.shape)
# print('---------------------')

time_data_load_end = time.time()

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

epochs = 40
batch_size = 64
dropout = 0.25
lr = 0.005
decay = 0.05
NUM_HEADS_ez=1
NUM_LAYERS_ez=1
NUM_HEADS_aal=1
NUM_LAYERS_aal=1
NUM_HEADS_cc400=4
NUM_LAYERS_cc400=1
NUM_HEADS_multi=8
NUM_LAYERS_multi=3
model_dim = 64
feature_dim = 64
fusion_dim = 64
margin = 11
alpha=2.5
beta=0.004
input_dim1 = Nodes_Atlas1 * Nodes_Atlas1
# input_dim2 = Nodes_Atlas2 * Nodes_Atlas2
input_dim3 = Nodes_Atlas3 * Nodes_Atlas3
# input_dim4 = Nodes_Atlas4 * Nodes_Atlas4
input_dim5 = Nodes_Atlas5 * Nodes_Atlas5
# input_dim6 = Nodes_Atlas6 * Nodes_Atlas6
# input_dim7 = Nodes_Atlas7 * Nodes_Atlas7





'''
    模型训练
'''
from sklearn.model_selection import KFold
time_train_list = []
for ind in range(1):
    setup_seed(ind)

    nodes_number1 = Nodes_Atlas1
    nums1 = np.ones(Nodes_Atlas1)
    nums1[:Nodes_Atlas1 - nodes_number1] = 0
    np.random.seed(1)
    np.random.shuffle(nums1)
    Mask = nums1.reshape(nums1.shape[0], 1) * nums1
    Masked_X1 = X1 * Mask
    J = nodes_number1
    for i in range(0, J):
        ind = i
        if nums1[ind] == 0:
            for j in range(J, Nodes_Atlas1):
                if nums1[j] == 1:
                    Masked_X1[:, [ind, j], :] = Masked_X1[:, [j, ind], :]
                    Masked_X1[:, :, [ind, j]] = Masked_X1[:, :, [j, ind]]
                    J = j + 1
                    break

    Masked_X1 = Masked_X1[:, :nodes_number1, :nodes_number1]
    X1 = Masked_X1

    acc_all = 0
    kf = KFold(n_splits=10, shuffle=True)
    kfold_index = 0
    for trainval_index, test_index in kf.split(X1, Y1):
        kfold_index += 1
        time_train_start = time.time()
        print('kfold_index:', kfold_index)

        X1_trainval, X1_test = X1[trainval_index], X1[test_index]
        Y1_trainval, Y1_test = Y1[trainval_index], Y1[test_index]
        # X2_trainval, X2_test = X2[trainval_index], X2[test_index]
        X3_trainval, X3_test = X3[trainval_index], X3[test_index]
        # X4_trainval, X4_test = X4[trainval_index], X4[test_index]
        X5_trainval, X5_test = X5[trainval_index], X5[test_index]
        # X6_trainval, X6_test = X6[trainval_index], X6[test_index]
        # X7_trainval, X7_test = X7[trainval_index], X7[test_index]

        Y_trainval, Y_test = Y1[trainval_index], Y1[test_index]

        for train_index, val_index in kf.split(X1_trainval, Y1_trainval):
            # 取消验证集
            X1_train, X1_val = X1_trainval[:], X1_trainval[:]
            # X2_train, X2_val = X2_trainval[:], X2_trainval[:]
            X3_train, X3_val = X3_trainval[:], X3_trainval[:]
            # X4_train, X4_val = X4_trainval[:], X4_trainval[:]
            X5_train, X5_val = X5_trainval[:], X5_trainval[:]
            # X6_train, X6_val = X6_trainval[:], X6_trainval[:]
            # X7_train, X7_val = X7_trainval[:], X7_trainval[:]
            Y_train, Y_val = Y1_trainval[:], Y1_trainval[:]
        # 统计信息，包括正例数量、负例数量
        print('X1_train', X1_train.shape)
        print('X1_test', X1_test.shape)
        print('Y1_train', Y_train.shape)
        print('Y1_test', Y_test.shape)
        print("Positive samples in training set: {}".format(np.sum(Y_train == 1)))
        print("Negative samples in training set: {}".format(np.sum(Y_train == 0)))
        print("Positive samples in test set: {}".format(np.sum(Y_test == 1)))
        print("Negative samples in test set: {}".format(np.sum(Y_test == 0)))
        model_aal = SimpleTransformerModel(Nodes_Atlas1 * Nodes_Atlas1, model_dim, num_classes=2,feature_dim=feature_dim, num_heads=NUM_HEADS_aal,num_layers=NUM_LAYERS_aal, dropout=dropout)
        # model_cc200 = SimpleTransformerModel(Nodes_Atlas2 * Nodes_Atlas2, model_dim, num_classes=2, feature_dim=feature_dim, dropout=dropout)
        model_cc400 = SimpleTransformerModel(Nodes_Atlas3 * Nodes_Atlas3, model_dim, num_classes=2, feature_dim=feature_dim, num_heads=NUM_HEADS_cc400,num_layers=NUM_LAYERS_cc400, dropout=dropout)
        # model_dos160 = SimpleTransformerModel(Nodes_Atlas4 * Nodes_Atlas4, model_dim, num_classes=2, feature_dim=feature_dim, dropout=dropout)
        model_ez = SimpleTransformerModel(Nodes_Atlas5 * Nodes_Atlas5, model_dim, num_classes=2, feature_dim=feature_dim, num_heads=NUM_HEADS_ez,num_layers=NUM_LAYERS_ez, dropout=dropout)
        # model_ho = SimpleTransformerModel(Nodes_Atlas6 * Nodes_Atlas6, model_dim, num_classes=2, feature_dim=feature_dim, dropout=dropout)
        # model_tt = SimpleTransformerModel(Nodes_Atlas7 * Nodes_Atlas7, model_dim, num_classes=2, feature_dim=feature_dim, dropout=dropout)

        # models = [model_aal, model_cc200, model_cc400, model_dos160, model_ez, model_ho, model_tt]
        # models = [model_aal, model_cc200, model_cc400]
        # models=[model_aal]
        # models = [model_aal, model_cc400]
        models = [model_aal, model_cc400, model_ez]
        multi_modal_model = MultiModalBrainModel(models, fusion_dim=fusion_dim,feature_dim=feature_dim, num_classes=2, model_dim=model_dim, num_heads=NUM_HEADS_multi, num_layers=NUM_LAYERS_multi, dropout=dropout)

        multi_modal_model.to(device)
        optimizer = optim.SGD(multi_modal_model.parameters(), lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
        # optimizer = optim.Adam(multi_modal_model.parameters(), lr=lr, weight_decay=decay)
        # optimizer = optim.RMSprop(multi_modal_model.parameters(), lr=lr, weight_decay=decay)
        # optimizer = optim.Adam(multi_modal_model.parameters(), lr=lr, weight_decay=decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=7, eta_min=0.001)

        loss_fn = nn.CrossEntropyLoss()
        infoNCE_loss_fn = InfoNCELoss()
        
        for epoch in range(1, epochs + 1):
            multi_modal_model.train()

            idx_batch = np.random.permutation(int(X1_train.shape[0]))
            num_batch = X1_train.shape[0] // int(batch_size)

            loss_train = 0
            loss_sim = 0
            loss_diff = 0
            loss_class = 0
            for bn in range(num_batch):
                if bn == num_batch - 1:
                    batch = idx_batch[bn * int(batch_size):]
                else:
                    batch = idx_batch[bn * int(batch_size): (bn + 1) * int(batch_size)]
                X1_batch = X1_train[batch].reshape(X1_train[batch].shape[0],-1)
                # X2_batch = X2_train[batch].reshape(X2_train[batch].shape[0],-1)
                X3_batch = X3_train[batch].reshape(X3_train[batch].shape[0],-1)
                # X4_batch = X4_train[batch].reshape(X4_train[batch].shape[0],-1)
                X5_batch = X5_train[batch].reshape(X5_train[batch].shape[0],-1)
                # X6_batch = X6_train[batch].reshape(X6_train[batch].shape[0],-1)
                # X7_batch = X7_train[batch].reshape(X7_train[batch].shape[0],-1)
                Y_batch = Y_train[batch]

                # 转换为Tensor
                X1_batch = torch.from_numpy(X1_batch).float().to(device)
                # X2_batch = torch.from_numpy(X2_batch).float().to(device)
                X3_batch = torch.from_numpy(X3_batch).float().to(device)
                # X4_batch = torch.from_numpy(X4_batch).float().to(device)
                X5_batch = torch.from_numpy(X5_batch).float().to(device)
                # X6_batch = torch.from_numpy(X6_batch).float().to(device)
                # X7_batch = torch.from_numpy(X7_batch).float().to(device)
                Y_batch = torch.from_numpy(Y_batch).long().to(device)

                # 增加序列维度
                X1_batch = X1_batch.unsqueeze(1)
                # X2_batch = X2_batch.unsqueeze(1)
                X3_batch = X3_batch.unsqueeze(1)
                # X4_batch = X4_batch.unsqueeze(1)
                X5_batch = X5_batch.unsqueeze(1)
                # X6_batch = X6_batch.unsqueeze(1)
                # X7_batch = X7_batch.unsqueeze(1)

                # train_data_batch = X1_train[batch]
                # train_label_batch = Y_train[batch]
                # train_data_batch = train_data_batch.reshape(train_data_batch.shape[0], -1)
                # train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                # train_data_batch_dev = train_data_batch_dev.unsqueeze(1)  # 增加序列维度
                # train_label_batch_dev = torch.from_numpy(train_label_batch).long().to(device)

                optimizer.zero_grad()
                # outputs, rec = model(train_data_batch_dev)
                # outputs = multi_modal_model(train_data_batch_dev)
                # outputs = multi_modal_model([X1_batch, X2_batch, X3_batch, X4_batch, X5_batch, X6_batch, X7_batch])
                # outputs = multi_modal_model([X1_batch, X2_batch, X3_batch])
                # outputs = multi_modal_model([X1_batch])
                outputs,similarity_output, difference_output = multi_modal_model([X1_batch, X3_batch,X5_batch])
                output_aal = model_aal(X1_batch)
                output_cc400 = model_cc400(X3_batch)
                output_ez = model_ez(X5_batch)

                # loss1 = loss_fn(outputs, train_label_batch_dev)
                # loss = loss1
                # loss = loss_fn(outputs, Y_batch)

                class_loss = loss_fn(outputs, Y_batch)  # 分类损失
                # sim_loss = similarity_loss([model_aal(X1_batch),model_cc400(X3_batch),model_ez(X5_batch)], alpha)  # 相似性损失
                # contrast_loss_aal_cc400 = contrastive_loss(output_aal, output_cc400)
                # contrast_loss_aal_ez = contrastive_loss(output_aal, output_ez)
                # contrast_loss_cc400_ez = contrastive_loss(output_cc400, output_ez)
                
                # 总对比损失
                # total_contrast_loss = beta*(contrast_loss_aal_cc400 + contrast_loss_aal_ez + contrast_loss_cc400_ez)

                # 假设 output_cc400 作为 anchor，output_ez 作为 positive，output_aal 作为 negative
                # total_contrast_loss = beta*infoNCE_loss_fn(output_aal, output_cc400, output_ez)
                total_contrast_loss = beta*infoNCE_loss_fn(output_cc400, output_ez, output_aal)

                # 相似性损失
                sim_loss = similarity_loss([output_aal, output_cc400, output_ez], alpha)
                
                
                # total_loss = class_loss + sim_loss  # 总损失
                total_loss = class_loss + sim_loss + total_contrast_loss
                # total_loss = class_loss
                # print('class_loss shape:', class_loss.shape)
                # print('sim_loss shape:', sim_loss.shape)
                # print('class_loss:', class_loss.item(), 'sim_loss:', sim_loss.item())
                # print('total_loss:', total_loss.item())
                
                loss_train += total_loss
                loss_class += class_loss
                loss_sim += sim_loss
                loss_diff += total_contrast_loss
                total_loss.backward()
                # loss.backward()
                optimizer.step()

            loss_train /= num_batch
            if epoch % 5 == 0:
                print('epoch:', epoch, 'train loss:', loss_train.item(),'class_loss:',loss_class.item(), 'sim loss:', loss_sim.item(), 'diff loss:', loss_diff.item())
            scheduler.step()
            # val
            if epoch % 1 == 0:
                multi_modal_model.eval()
                # 测试数据处理
                X1_test_batch = X1_test.reshape(X1_test.shape[0],-1)
                # X2_test_batch = X2_test.reshape(X2_test.shape[0],-1)
                X3_test_batch = X3_test.reshape(X3_test.shape[0],-1)
                # X4_test_batch = X4_test.reshape(X4_test.shape[0],-1)
                X5_test_batch = X5_test.reshape(X5_test.shape[0],-1)
                # X6_test_batch = X6_test.reshape(X6_test.shape[0],-1)
                # X7_test_batch = X7_test.reshape(X7_test.shape[0],-1)
                Y1_test_batch = Y1_test
                # 转换为Tensor
                X1_test_batch = torch.from_numpy(X1_test_batch).float().to(device)
                # X2_test_batch = torch.from_numpy(X2_test_batch).float().to(device)
                X3_test_batch = torch.from_numpy(X3_test_batch).float().to(device)
                # X4_test_batch = torch.from_numpy(X4_test_batch).float().to(device)
                X5_test_batch = torch.from_numpy(X5_test_batch).float().to(device)
                # X6_test_batch = torch.from_numpy(X6_test_batch).float().to(device)
                # X7_test_batch = torch.from_numpy(X7_test_batch).float().to(device)
                Y1_test_batch = torch.from_numpy(Y1_test_batch).long().to(device)

                # 增加序列维度
                X1_test_batch = X1_test_batch.unsqueeze(1)
                # X2_test_batch = X2_test_batch.unsqueeze(1)
                X3_test_batch = X3_test_batch.unsqueeze(1)
                # X4_test_batch = X4_test_batch.unsqueeze(1)
                X5_test_batch = X5_test_batch.unsqueeze(1)
                # X6_test_batch = X6_test_batch.unsqueeze(1)
                # X7_test_batch = X7_test_batch.unsqueeze(1)

                # test_data_batch = X_test.reshape(X_test.shape[0], -1)  # 重塑为 (num_test_samples, 40000)
                # test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                # test_data_batch_dev = test_data_batch_dev.unsqueeze(1)  # 添加伪序列维度 (num_test_samples, 1, 40000)
                # test_data_batch_dev = torch.from_numpy(X_test).float().to(device)
                # outputs, _= model(test_data_batch_dev)
                # outputs = multi_modal_model(test_data_batch_dev)
                # outputs = multi_modal_model([X1_test_batch, X2_test_batch, X3_test_batch, X4_test_batch, X5_test_batch, X6_test_batch, X7_test_batch])
                # outputs = multi_modal_model([X1_test_batch, X2_test_batch, X3_test_batch])
                # outputs = multi_modal_model([X1_test_batch])
                outputs,similarity_output, difference_output = multi_modal_model([X1_test_batch, X3_test_batch,X5_test_batch])
                _, indices = torch.max(outputs, dim=1)
                preds = indices.cpu()
                acc = metrics.accuracy_score(preds, Y1_test)
                # recall = recall_score(preds, Y_test, average='macro')  # 可以根据需要选择average参数
                # f1 = f1_score(preds, Y_test, average='macro')  # 可以根据需要选择average参数
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 假设第二列是正类的概率
                probs_list = probs.detach().cpu().numpy()
                # y_true_list = Y1_test.cpu().numpy()

                # 检查y_true中是否有多于一个类别
                if len(np.unique(Y1_test)) > 1:
                    auc_score = roc_auc_score(Y1_test, probs_list)
                    auc_list.append(auc_score)
                else:
                    # 处理只有一个类别的情况
                    # print(f"Only one class present in fold {kfold_index}. AUC not defined.")
                    # 选择性地添加默认值，例如
                    auc_list.append(0.5)  # 或者其他您认为合适的默认值
                # auc = roc_auc_score(preds, Y1_test)
                recall = recall_score(preds, Y1_test, average='macro', zero_division=1)
                f1 = f1_score(preds, Y1_test, average='macro', zero_division=1)
                print('Test acc', acc, 'Test recall', recall, 'Test f1', f1, 'Test auc', auc_score)
                recall_list.append(recall)
                f1_list.append(f1)


        torch.save(multi_modal_model.state_dict(), './modelstfensemble8/' + str(kfold_index) + '.pt')
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
print(f"Data Loading Time: {time_data_load_end - time_data_load_start} seconds")
print(f"Ave Training Time: {np.mean(time_train_list)} seconds")
# 将result、recall_k、f1_k、auc_k保存到同一个文件result.csv中，纵列为kfold编号，横列标号分别为acc、recall、f1、AUC


'''
    模型测试
'''




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