# GraphSAGE

### 前言

此文提出的方法叫GraphSAGE，针对的问题是之前的网络表示学习的transductive，从而提出了一个inductive的GraphSAGE算法。GraphSAGE同时利用节点特征信息和结构信息得到Graph Embedding的映射，相比之前的方法，之前都是保存了映射后的结果，而GraphSAGE保存了生成embedding的映射，可扩展性更强，对于节点分类和链接预测问题的表现也比较突出。

现存的方法需要图中所有的顶点在训练embedding的时候都出现；这些前人的方法本质上是transductive，不能自然地泛化到未见过的顶点。

文中提出了GraphSAGE，是一个inductive的框架，可以利用顶点特征信息（比如文本属性）来高效地为没有见过的顶点生成embedding。

GraphSAGE是为了学习一种节点表示方法，即如何通过从一个顶点的局部邻居采样并聚合顶点特征，而不是为每个顶点训练单独的embedding。

### 1.GCN以及问题

GCN虽然能提取图中顶点的embedding，但是存在一些问题：
**GCN的基本思想**： 把一个节点在图中的高纬度邻接信息降维到一个低维的向量表示。
**GCN的优点**： 可以捕捉graph的全局信息，从而很好地表示node的特征。
**GCN的缺点**： Transductive learning的方式，需要把所有节点都参与训练才能得到node embedding，无法快速得到新node的embedding。

**transductive learning vs inductive learning**

前人的工作专注于从一个固定的图中对顶点进行表示，很多现实中的应用需要很快的对未见过的顶点或是全新的图（子图）生成embedding**。**这种推断的能力对于高吞吐的机器学习系统来说很重要，这些系统都运作在不断演化的图上，而且时刻都会遇到未见过的顶点（比如Reddit上的帖子（posts），Youtube上的用户或视频）。因此，一种inductive的学习方法比transductive的更重要。

**transductive learning得到新节点的表示的难处**：
要想得到新节点的表示，需要让新的graph或者subgraph去和已经优化好的node embedding去“对齐（align）”。然而每个节点的表示都是受到其他节点的影响，因此添加一个节点，意味着许许多多与之相关的节点的表示都应该调整。这会带来极大的计算开销，即使增加几个节点，也要完全重新训练所有的节点。

**GraphSAGE基本思路**：
既然新增的节点，一定会改变原有节点的表示，那么为什么一定要得到每个节点的一个固定的表示呢？何不**直接学习一种节点的表示方法**。去学习一个节点的信息是怎么通过其邻居节点的特征聚合而来的。 学习到了这样的“聚合函数”，而我们本身就已知各个节点的特征和邻居关系，我们就可以很方便地得到一个新节点的表示了。

GCN等transductive的方法，学到的是每个节点的一个唯一确定的embedding； 而GraphSAGE方法学到的node embedding，是根据node的邻居关系的变化而变化的，也就是说，即使是旧的node，如果建立了一些新的link，那么其对应的embedding也会变化，而且也很方便地学到。

本文出自斯坦福PinSAGE的理论篇，关于第一个基于GCN的工业级推荐系统的PinSAGE可以看这篇：
[[PinSage\] Graph Convolutional Neural Networks for Web-Scale Recommender Systems 论文详解KDD2018](https://blog.csdn.net/yyl424525/article/details/100986283)

### 2.相关工作

GraphSAGE算法在概念上与以前的节点embedding方法、一般的图形学习监督方法以及最近将卷积神经网络应用于图形结构化数据的进展有关。

Factorization-based embedding approaches（节点embedding）

一些node embedding方法使用随机游走的统计方法和基于矩阵分解学习目标学习低维的embeddings

- Grarep: Learning graph representations with global structural information. In KDD, 2015
- node2vec: Scalable feature learning for networks. In KDD, 2016
- Deepwalk: Online learning of social representations. In KDD, 2014
- Line: Large-scale information network embedding. In WWW, 2015
- Structural deep network embedding. In KDD, 2016

这些embedding算法直接训练单个节点的节点embedding，本质上是transductive，而且需要大量的额外训练（如随机梯度下降）使他们能预测新的顶点。

此外，Yang et al.的Planetoid-I算法，是一个inductive的基于embedding的半监督学习算法。然而，Planetoid-I在推断的时候不使用任何图结构信息，而在训练的时候将图结构作为一种正则化的形式。

不像前面的这些方法，本文利用特征信息来训练可以对未见过的顶点生成embedding的模型。

Supervised learning over graphs

Graph kernel
除了节点嵌入方法，还有大量关于图结构数据的监督学习的文献。这包括各种各样的基于内核的方法，其中图的特征向量来自不同的图内核(参见Weisfeiler-lehman graph kernels和其中的引用)。

一些神经网络方法用于图结构上的监督学习，本文的方法在概念上受到了这些算法的启发

- Discriminative embeddings of latent variable models for structured data. In ICML, 2016
- A new model for learning in graph domains
- Gated graph sequence neural networks. In ICLR, 2015
- The graph neural network model

然而，这些以前的方法是尝试对整个图(或子图)进行分类的，但是本文的工作的重点是为单个节点生成有用的表示。

Graph convolutional networks

近年来，提出了几种用于图上学习的卷积神经网络结构

- Spectral networks and locally connected networks on graphs. In ICLR, 2014
- Convolutional neural networks on graphs with fast localized spectral filtering. In NIPS, 2016
- Convolutional networks on graphs for learning molecular fingerprints. In NIPS,2015
- Semi-supervised classification with graph convolutional networks. In ICLR, 2016
- Learning convolutional neural networks for graphs. In ICML, 2016

这些方法中的大多数不能扩展到大型图，或者设计用于全图分类(或者两者都是)。

GraphSAGE可以看作是对transductive的GCN框架对inductive下的扩展。

### 3.GraphSAGE原理

GraphSAGE的核心：GraphSAGE不是试图学习一个图上所有node的embedding，而是学习一个为每个node产生embedding的映射。

文中不是对每个顶点都训练一个单独的embeddding向量，而是训练了一组aggregator functions，这些函数学习如何从一个顶点的局部邻居聚合特征信息（见图1）。**每个聚合函数从一个顶点的不同的hops或者说不同的搜索深度聚合信息**。测试或是推断的时候，**使用训练好的系统，通过学习到的聚合函数来对完全未见过的顶点生成embedding**。

![img](https://gitee.com/karlhan/picgo/raw/master/img//WEB8531ef13d2695c2bffed2d915ec14f0f)

GraphSAGE 是Graph SAmple and aggreGatE的缩写，其运行流程如上图所示，可以分为三个步骤：

- 对图中每个顶点邻居顶点进行采样，因为每个节点的度是不一致的，为了计算高效， 为每个节点采样固定数量的邻居
- 根据聚合函数聚合邻居顶点蕴含的信息
- 得到图中各顶点的向量表示供下游任务使用

文中设计了无监督的损失函数，使得GraphSAGE可以在没有任务监督的情况下训练。实验中也展示了使用完全监督的方法如何训练GraphSAGE。

**3.1 Embedding generation  algorithm 生成节点embedding的前向传播算法**

前向传播描述了如何使用聚合函数对节点的邻居信息进行聚合，从而生成节点embedding：

![image-20200909130717874](https://gitee.com/karlhan/picgo/raw/master/img//image-20200909130717874.png)

**在每次迭代(或搜索深度)，顶点从它们的局部邻居聚合信息，并且随着这个过程的迭代，顶点会从越来越远的地方获得信息**。PinSAGE使用的前向传播算法和GraphSAGE一样，GraphSAGE是PinSAGE的理论基础。

算法1描述了在整个图上生成embedding的过程，其中

![image-20200909130852087](https://gitee.com/karlhan/picgo/raw/master/img//image-20200909130852087.png)

为了将算法1扩展到minibatch环境上，给定一组输入顶点，先采样采出需要的邻居集合（直到深度KK*K*），然后运行内部循环（算法1的第三行）（附录A包括了完整的minibatch伪代码）。

**文中在较大的数据集上实验。因此，统一采样一个固定大小的邻域集，以保持每个batch的计算占用空间是固定的（即 graphSAGE并不是使用全部的相邻节点，而是做了固定size的采样）**。

**这样固定size的采样，每个节点和采样后的邻居的个数都相同，可以把每个节点和它们的邻居拼成一个batch送到GPU中进行批训练。**

![image-20200909131405965](https://gitee.com/karlhan/picgo/raw/master/img//image-20200909131405965.png)

- **这里需要注意的是，每一层的node的表示都是由上一层生成的，跟本层的其他节点无关，这也是一种基于层的采样方式**。
- 在图中的“1层”，节点v聚合了“0层”的两个邻居的信息，v的邻居u也是聚合了“0层”的两个邻居的信息。到了“2层”，可以看到节点v通过“1层”的节点u，扩展到了“0层”的二阶邻居节点。因此，在聚合时，聚合K次，就可以扩展到K阶邻居。
- **没有这种采样，单个batch的内存和预期运行时是不可预测的，在最坏的情况下是O(∣V∣)**。相比之下，GraphSAGE的每个batch空间和时间复杂度是固定的
- **实验发现，K不必取很大的值，当K=2时，效果就很好了**。至于邻居的个数，文中提到S1⋅S2≤500，即两次扩展的邻居数之际小于500，**大约每次只需要扩展20来个邻居**时获得较高的性能。
- 论文里说固定长度的随机游走其实就是随机选择了固定数量的邻居

**3.2 聚合函数的选取**

在图中顶点的邻居是无序的，所以希望构造出的聚合函数是对称的（即也就是对它输入的各种排列，函数的输出结果不变），同时具有较高的表达能力。 聚合函数的对称性（symmetry property）确保了神经网络模型可以被训练且可以应用于任意顺序的顶点邻居特征集合上。

##### Mean aggregator

mean aggregator将目标顶点和邻居顶点的第k−1层向量拼接起来，然后对向量的每个维度进行求均值的操作，将得到的结果做一次非线性变换产生目标顶点的第kk*k*层表示向量。
文中用下面的式子替换算法1中的4行和5行**得到GCN的inductive变形**：

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200909132733256.png" alt="image-20200909132733256" style="zoom:67%;" />

- 均值聚合近似等价在transducttive GCN框架[Semi-supervised classification with graph convolutional networks. In ICLR, 2016]中的卷积传播规则
- 文中称这个修改后的基于均值的聚合器是convolutional的，这个卷积聚合器和文中的其他聚合器的重要不同在于**它没有算法1中第5行的CONCAT操作**——卷积聚合器没有将顶点前一层的表示 $h^{k-1}_{v}$和聚合的邻居向量$h^k_{N(v)}$拼接起来。
- 拼接操作可以看作一个是GraphSAGE算法在不同的搜索深度或层之间的简单的**skip connection**[Identity mappings in deep residual networks]的形式，它使得模型获得了巨大的提升
- 举个简单例子，比如一个节点的3个邻居的embedding分别为[1,2,3,4],[2,3,4,5],[3,4,5,6]按照每一维分别求均值就得到了聚合后的邻居embedding为[2,3,4,5]

##### LSTM aggregator





























