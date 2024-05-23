# NEU_MultiModalBrainModel
基于功能脑网络多模版学习的精神疾病辅助诊断算法研究

我阅读了大量关于功能脑网络分析、多模态学习、图嵌入学习、以及神经科学领域精神疾病的诊断研究文献。

不同的脑模版生成的脑网络的主要区别在于脑区划分的粒度不同，导致在不同脑模板上有多种视角的信息表达，因此多模版脑网络学习实际上是一种多视图学习。以往的工作可分为两类。传统的机器学习方法独立的抽取各个视图的特征之后用于疾病分类，忽略了不同视图之间的关联和信息冗余。另一类工作主要采用图嵌入学习方法深度挖掘脑网络特征，通过最大限度地提高不同视图之间的相互一致性，从而捕捉不同视图之间的关联，但不同视图之间高度冗余信息并没有得以处理，影响了信息融合。因此，目前急需一种有效的手段充分有效融合多视图图信息。

我结合集成学习策略构建了一个融合了多个模型的MultiModalBrainModel。首先高效的从原脑模板数据中提取特征，我实验了三个编码器。如果一个编码器单独应用能够高效的分类，说明它能够很好地提取特征。

我实验了3个编码器，分别是基于边卷积的E2EModel，一个我自己的从超图理论获得灵感的SimpleTransformerModel，以及一个融合了图神经网络(GNN)和Transformer的SGFormer。在ABIDE(Autism Brain Imaging Data Exchange)数据集上进行了实验，该数据集包含了丰富的自闭症谱系障碍(ASD)患者与健康对照组的脑成像数据，经过处理后得到了脑区之间的连通性数据。

| 编码器	|ACC	|Time|
| :---: | :---: | :---: |
| E2EModel |	0.642881 | 	71.13 |
| SimpleTransformer |	0.677273 |	3.49 |
| SGFormer |	0.536181 | 	89.25 |

如上表所示，SimplerTransformer的时间短、效果好，因此后面的多模板模型将基于SimpleTransformer。这个模型能够捕获输入数据的长距离依赖关系，适合于处理超节点间的复杂关系。

之后在不同的脑图谱上进行了独立的训练和十折交叉验证，寻找信息容量较大的视图，据此选定了相对重要的视图(aal, cc400, ez)。

基于E2E编码器
![E2E](https://github.com/quantumxiaol/NEU_MultiModalBrainModel/blob/main/png/E2E.png "E2E")

SimpleTransormer编码器
![SimpleTransormer](https://github.com/quantumxiaol/NEU_MultiModalBrainModel/blob/main/png/simpleTransormer.png "SimpleTransormer")

SGFormer编码器
![SGFormer](https://github.com/quantumxiaol/NEU_MultiModalBrainModel/blob/main/png/SGFormer.png "SGFormer")

集成模型
![MultiModalBrain](https://github.com/quantumxiaol/NEU_MultiModalBrainModel/blob/main/png/MultiModalBrain.png "MultiModalBrain")

开发MultiModalBrainModel是为了试图利用不同粒度下的脑区划分带来的多视角信息。这个模型利用了多个子模型来从原始脑区连通度捕获特征，Transformer Encoder来提取和融合特征，最终用于分类任务。模型的结构设计用于捕捉不同模态间的相似性和差异性，有助于处理结构性和功能性脑成像数据。该模型包含了两个编码器，分别从相似性和差异性的角度学习不同视图间的特征，通过分类损失和相似性损失的结合来更新编码器。使用CosineSimilarity作为相似性损失函数，探索不同视图中的共同疾病相关特征。采用InfoNCE作为差异性损失函数，寻找不同视图的差异。

| 模型 |	ACC |	Recall |	F1score |	AUC |
| :---: | :---: | :---: | :---: | :---: |
| MultiModalBrainModel |	0.692385 |	0.680739 |	0.671735 |	0.729583 |
| SimplerTransformer(cc400) |	0.686507 |	0.686950 |	0.681376 |	0.756921 |
| SimplerTransformer(aal) |	0.622309 |	0.635767 |	0.626127 |	0.688950 |

目前在独立的训练和十折交叉验证中平均准确率(ACC)为0.692385，平均召回率(Recall)为0.680739，平均F1score为0.671735，平均ROC曲线下与坐标轴围成的面积(AUC)为0.729583，这比单一的cc400和aal表现好，表明模型能够利用多视图的信息，同时具有更好的鲁棒性和泛化能力。
