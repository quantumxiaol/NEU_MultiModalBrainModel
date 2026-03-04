# NEU_MultiModalBrainModel

基于功能脑网络多模版学习的精神疾病辅助诊断算法研究

我阅读了大量关于功能脑网络分析、多模态学习、图嵌入学习、以及神经科学领域精神疾病的诊断研究文献。

不同的脑模版生成的脑网络的主要区别在于脑区划分的粒度不同，导致在不同脑模板上有多种视角的信息表达，因此多模版脑网络学习实际上是一种多视图学习。以往的工作可分为两类。传统的机器学习方法独立的抽取各个视图的特征之后用于疾病分类，忽略了不同视图之间的关联和信息冗余。另一类工作主要采用图嵌入学习方法深度挖掘脑网络特征，通过最大限度地提高不同视图之间的相互一致性，从而捕捉不同视图之间的关联，但不同视图之间高度冗余信息并没有得以处理，影响了信息融合。因此，目前急需一种有效的手段充分有效融合多视图图信息。

## 数据集

在脑网络研究中，常用的脑区划分模板有多个，包括 AAL（Automated Anatomical Labeling）、CC200（Craddock 200）、CC400（Craddock 400）、DOS160（Dosenbach 160）、EZ（Eickhoff-Zilles）、HO（Harvard-Oxford） 和 TT（Talairach-Tournoux）等。

AAL模板是一种基于结构的脑区划分方法，将脑分为116个区域。该模板广泛应用于功能和结构连接研究。AAL模板基于结构MRI数据，提供标准化的脑区划分，适用于多种脑功能和连接性研究。

CC200模板是一种基于功能的脑区划分方法，通过聚类分析将脑分为200个功能连接区域。CC200模板通过聚类算法基于rs-fMRI数据生成，强调功能连接的同质性，适用于功能连接研究。

CC400模板是CC200的扩展版本，通过更细粒度的聚类将脑分为400个功能连接区域。CC400模板提供了更高分辨率的功能连接划分，适用于需要精细功能区分的研究。

DOS160模板基于Dosenbach等人提出的一种包含160个ROI的功能连接模板，主要用于任务态和静息态功能连接研究。DOS160模板整合了多个功能网络，适用于跨任务和静息态的功能连接分析。

EZ模板基于Eickhoff和Zilles提出的脑区划分方法，广泛应用于解剖和功能研究。本研究的脑区数量为116。EZ模板提供高分辨率的解剖学细节，适用于精细的脑结构和功能研究。

HO模板由Harvard大学和Oxford大学联合开发，基于高分辨率的结构MRI数据，将脑分为96个区域。HO模板提供高精度的解剖学划分，广泛应用于结构和功能研究。

TT模板基于Talairach和Tournoux的脑区坐标系统，提供标准化的解剖学脑区划分。本研究的脑区数量为97。TT模板是经典的脑区坐标系统，适用于标准化的脑成像研究。

本研究基于ABIDE，该数据集是一个公共自闭症（ASD）研究数据库，汇集了来自17个不同采集点的1112名受试者的rs-fMRI和表型数据。在这项工作中，利用Connectome Computing System对图像进行预处理。预处理包括切片定时校正、运动校正和体素强度归一化等。本研究共纳入871名优质受试者，包括403名ASD患者（女性54名，男性349名，年龄17.07±7.96岁，范围7-58岁）和468名正常对照（女性90名，男性378名，年龄16.84±7.24岁，范围6-56岁。

数据为反映脑区之间连通性的图。如有200个脑区，图的大小为(200,200)，存放的mat为(871,200,200)。

下图为一个病人的cc200热图。
![cc200](./png/cc200.png "cc200")

## 模型策略

我结合集成学习策略构建了一个融合了多个模型的MultiModalBrainModel。首先高效的从原脑模板数据中提取特征，我实验了三个编码器。如果一个编码器单独应用能够高效的分类，说明它能够很好地提取特征。

我实验了3个编码器，分别是基于边卷积的E2EModel，一个我自己的从超图理论获得灵感的SimpleTransformerModel，以及一个融合了图神经网络(GNN)和Transformer的SGFormer。在ABIDE(Autism Brain Imaging Data Exchange)数据集上进行了实验，该数据集包含了丰富的自闭症谱系障碍(ASD)患者与健康对照组的脑成像数据，经过处理后得到了脑区之间的连通性数据。

编码器的实验[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1itucZ6gqmhtzutG-fLSuotDHyweQstYc)

注意到SGFormer依赖的torch-sparse不太好装，先跳过这个部分。

我自己的是RTX4080LaptopGPU（12GB）,下面的时间是在此设备上的计算时间。

| 编码器	|ACC	|Time(s) | 参数量 (CC200) | 计算量 (CC200) |
| :---: | :---: | :---: | :---: | :---: |
| E2EModel |	0.642881 | 	71.13 | 2.04M | 8.00M FLOPs |
| SimpleTransformer |	0.677273 |	3.49 | 2.61M | 3.38M FLOPs |
| SGFormer |	0.536181 | 	89.25 | 0.033M | 6.40M FLOPs |

如上表所示，SimplerTransformer的时间短、效果好，因此后面的多模板模型将基于SimpleTransformer。这个模型能够捕获输入数据的长距离依赖关系，适合于处理超节点间的复杂关系。

之后在不同的脑图谱上进行了独立的训练和十折交叉验证，寻找信息容量较大的视图，据此选定了相对重要的视图(aal, cc400, ez)。

基于E2E编码器
![E2E](./png/E2E.png "E2E")

SimpleTransormer编码器
![SimpleTransormer](./png/simpleTransormer.png "SimpleTransormer")

SGFormer编码器
![SGFormer](./png/SGFormer.png "SGFormer")

集成模型
![MultiModalBrain](./png/MultiModalBrain.png "MultiModalBrain")

开发MultiModalBrainModel是为了试图利用不同粒度下的脑区划分带来的多视角信息。这个模型利用了多个子模型来从原始脑区连通度捕获特征，Transformer Encoder来提取和融合特征，最终用于分类任务。模型的结构设计用于捕捉不同模态间的相似性和差异性，有助于处理结构性和功能性脑成像数据。该模型包含了两个编码器，分别从相似性和差异性的角度学习不同视图间的特征，通过分类损失和相似性损失的结合来更新编码器。使用CosineSimilarity作为相似性损失函数，探索不同视图中的共同疾病相关特征。采用InfoNCE作为差异性损失函数，寻找不同视图的差异。

集成模型的表现[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JcMVQw3rHGiJX5rQpQXWq352HV26RpR7)

## 结果

| 模型 |	ACC |	Recall |	F1score |	AUC | 参数量 | 计算量 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MultiModalBrainModel |	0.692385 |	0.680739 |	0.671735 |	0.729583 | 12.31M | 16.59M FLOPs |
| SimplerTransformer(cc400) |	0.686507 |	0.686950 |	0.681376 |	0.756921 | 5.13M | 5.53M FLOPs |
| SimplerTransformer(aal) |	0.622309 |	0.635767 |	0.626127 |	0.688950 | 0.91M | 1.34M FLOPs |

目前在独立的训练和十折交叉验证中平均准确率(ACC)为0.692385，平均召回率(Recall)为0.680739，平均F1score为0.671735，平均ROC曲线下与坐标轴围成的面积(AUC)为0.729583，这比单一的cc400和aal表现好，表明模型能够利用多视图的信息，同时具有更好的鲁棒性和泛化能力。

## 论文

论文参见paper/paper.tex。模版和学校要求的不同。我也不想写这么啰嗦，本来几千字能说清楚的，论文要求20000字，还有图表，还有参考文献的个数。我只能灌水，加用到的模型的理论基础，加其他机器学习的对比实验。

还有这个题目，沾了“算法”两个字，就要详尽分析时间复杂度和空间复杂度。模型似乎更多用的是参数量和计算量。

沾了“多模版”，当时学长告诉我这里模版就是不同划分的脑图谱，可以类比当成多模态来。结果答辩时我被逮着模版和模态的区别问。还有论文的字体问题。公式排版问题。本来验收代码成果时还交流的很好，看来是论文表现和代码落差太大。“不是我针对你，是你这个论文写的实在是……，我昨天还高兴呢，结果今天你就这样”。

但是话又说回来，有的人答辩前还没装好python运行环境，有的人项目任务没完成，它们就什么事都没有。论文写的怎么样先不谈，至少我敢公开训练数据、运行环境、项目代码、超参数配置。

```bash
cd paper
xelatex paper.tex
bibtex paper
xelatex paper.tex
xelatex paper.tex
```
编译论文。

## 本地运行

```bash
git clone https://github.com/quantumxiaol/NEU_MultiModalBrainModel.git

cd NEU_MultiModalBrainModel

# 下载数据集，放到ABIDEdata下

uv lock

uv sync

# 配置计算设备
cat .env.template > .env

# 运行BrainADL_ABIDE开头的python文件，单个文件可以独立执行，后缀表明了脑图和模型
```

我完成原始的任务在Windows上运行的，设备为intel i9-13900HX+nVidia RTX 4080 Laptop，在WSL、ubuntu上验证了可以运行。后面更新了环境，确认了在mac OS（M4 Pro）上可以使用mps计算。

## 后记

改这个早已完结的项目，就像柯西展示了他关于级数收敛性的新理论后，拉普拉斯拿着柯西的判别法，逐一检查《天体力学》中用到的每一个级数。

Transformer 的计算，出现序列长度坍缩（Sequence Length Collapse）导致自注意力机制失效。
模型实际上变成了一个拥有 40,000 个输入维度的深层 MLP。
40,000 维的特征空间对于 871 个样本来说太“宽阔”了。模型通过极其复杂的非线性映射，轻而易举地在训练集里画出了分类界线。
acc为0.677273，4% 的提升让我觉得“Transformer 的全局建模能力起作用了”，但实际上，似乎这只是深度模型靠参数量硬堆出来的拟合能力。

我把Transformer的实现改过了，原来是“超图的节点”作为Token，现在是一个脑区对应一个Token。同时添加attention map的实现。

学长一开始给我的那个就有点问题，十折平均、取消验证集、只有训练集测试集，带有数据泄露的隐患。初衷是医学数据太少了，再划出去验证集会导致训练的数据更少。现在划分验证集，并采用早停。

鉴于只有871个数据，传统模型如RF、SVM自然会表现更好一点。

### 新的实验数据

cc200上不同模型的表现

|模型|cnn|simpletf|sgformer|
|-|-|-|-|
|acc|0.6543887147335424|0.5866118077324974|0.5326541274817137|
|time|65.907|10.72|138.06|


transformer在不同模版上的表现

|模版|aal|cc200|cc400|dos|ez|ho|tt|
|-|-|-|-|-|-|-|-|
|acc|0.5992816091954023|0.5866118077324974|0.6221525600835947|0.5658699059561129|0.6130485893416927|0.6107628004179728|0.591274817136886|
|time|18.03|10.72|32.5754|19.51|12.53|12.36|11.05|

cc400不同模型的表现

|模型|cnn|simpletf|Graphformer|SVM|RF|
|-|-|-|-|-|-|
|acc|0.5797283176593522|0.6221525600835947|0.6061128526645769|0.6877220480668758|0.6406217345872518
|Recall|0.5789179140639468|0.6175845711335227|0.6025647274745152|0.6807686136634227|0.6254032048316311
|F1|0.5492135948553313|0.6074994223101176|0.5822445560834322|0.6810094433921405|0.6120624153427312
|AUC|0.6409376097932221|0.6599171124896835|0.646700622967953|0.7480032103343304|0.6942460306639439|
time|258.03|24.1288|143.31|442.54|7.186|

cc400上Transformer的attention map。

![asd_mean_last_laye](./png/cc400tf_attention_map/asd_mean_last_layer_global.png "asd_mean_last_laye")
![hc_mean_last_layer](./png/cc400tf_attention_map/hc_mean_last_layer_global.png "hc_mean_last_layer")

以看到明显的纵向和横向条纹。某些特定的脑区（对应的 Index）充当了“信息枢纽（Hubs）”。模型在做最终决定时，会反复向这些脑区“提问”（Query）或者从它们那里“提取信息”（Key）。


![asd_mean_rollout](./png/cc400tf_attention_map/asd_mean_rollout_global.png "asd_mean_rollout")
![hc_mean_rollout](./png/cc400tf_attention_map/hc_mean_rollout_global.png "hc_mean_rollout")

有一条极亮的对角线。这代表自注意力（Self-attention）占主导，即模型认为每个脑区自身的连接特征（与全脑的 Profile）是识别疾病的最稳健特征。


![asd_minus_hc_last_layer](./png/cc400tf_attention_map/asd_minus_hc_last_layer_global.png "asd_minus_hc_last_layer")
![asd_minus_hc_rollout](./png/cc400tf_attention_map/asd_minus_hc_rollout_global.png "asd_minus_hc_rollout")

红色代表 ASD 比 HC 受到模型更多关注的连接，蓝色则相反。图中的红色斑点是模型眼中的“自闭症特异性连接”。


集成模型的表现
||cc400+aal+ez|
|-|-|
|acc|0.5981060606060605|
|Recall|0.5997984589697886|
|F1|0.5733345386475863|
|AUC|0.6734767942961576|
|time|36.392|

通过集成模型，AUC能力有了提升，说明确实提升了模型的泛化和分类能力。