# GRAPH ATTENTION NETWORKS

### 一、 前言

深度学习三巨头”之一的Yoshua Bengio组提出了Graph Attention Networks（下述简称为GAT）去解决GCN存在的问题并且在不少的任务上都取得了state of art的效果（可以参考[机器之心：深入理解图注意力机制](https://zhuanlan.zhihu.com/p/57180498)的复现结果），是graph neural network领域值得关注的工作。

<img src="https://pic3.zhimg.com/80/v2-a31700b686a553ce2daa7e077a84940f_720w.jpg" alt="img" style="zoom: 67%;" />

Graph数据结构的两种“特征”，指的是定点和边的关系。研究目标聚焦在顶点之上，边诉说着顶点之间的关系。**对于任意一个顶点 ![[公式]](https://www.zhihu.com/equation?tex=i) ，它在图上邻居 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BN%7D_i) ,构成第一种特征，即图的结构关系。**

当然，除了图的结构之外，每个顶点还有自己的特征 ![[公式]](https://www.zhihu.com/equation?tex=h_i) （通常是一个高维向量）。

##### GCN的局限性

GCN是处理transductive任务的一把利器（transductive任务是指：训练阶段与测试阶段都基于同样的图结构），然而GCN有**两大局限性**是经常被诟病的：

（a）无法完成inductive任务，即处理动态图问题。inductive任务是指：训练阶段与测试阶段需要处理的graph不同。通常是训练阶段只是在子图（subgraph）上进行，测试阶段需要处理未知的顶点。（unseen node）

（b）处理有向图的瓶颈，不容易实现分配不同的学习权重给不同的neighbor。这一点在前面的文章中已经讲过了，不再赘述，如有需要可以参考下面的链接。

### 二、GAT的结构

##### GAT的两种运算方式

- **Global graph attention**

就是每一个顶点 ![[公式]](https://www.zhihu.com/equation?tex=i) 都对于图上任意顶点都进行attention运算。可以理解为上图的蓝色顶点对于其余全部顶点进行一遍运算。

优点：完全不依赖于图的结构，对于inductive任务无压力

缺点：（1）丢掉了图结构的这个特征，无异于自废武功，效果可能会很差

​			（2）运算面临着高昂的成本

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200904150110402.png" alt="image-20200904150110402"  />

![image-20200904150243972](https://gitee.com/karlhan/picgo/raw/master/img//image-20200904150243972.png)

- **Mask graph attention**

注意力机制的运算只在邻居顶点上进行，也就是说图1的蓝色顶点只计算和橙色顶点的注意力系数。

作者在文中采用的是masked attention

##### GAT的计算

(1) 计算注意力系数(attention coefficient)

对于顶点 ![[公式]](https://www.zhihu.com/equation?tex=i) ，逐个计算它的邻居们（ ![[公式]](https://www.zhihu.com/equation?tex=j+%5Cin+%5Cmathcal%7BN%7D_i) ）和它自己之间的相似系数

![image-20200904150508591](https://gitee.com/karlhan/picgo/raw/master/img//image-20200904150508591.png)

首先一个共享参数 ![[公式]](https://www.zhihu.com/equation?tex=W) 的线性映射对于顶点的特征进行了增维，当然这是一种常见的特征增强（feature augment）方法；![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5B+%5Ccdot+%5Cbig%7C+%5Cbig%7C+%5Ccdot%5Cright%5D+) 对于顶点 ![[公式]](https://www.zhihu.com/equation?tex=i%2Cj) 的变换后的特征进行了拼接（concatenate）；最后 ![[公式]](https://www.zhihu.com/equation?tex=a%28%5Ccdot%29) 把拼接后的高维特征映射到一个实数上，作者是通过 single-layer feedforward neural network实现的。

显然学习顶点 ![[公式]](https://www.zhihu.com/equation?tex=i%2Cj) 之间的相关性，就是通过可学习的参数 ![[公式]](https://www.zhihu.com/equation?tex=W) 和映射 ![[公式]](https://www.zhihu.com/equation?tex=a%28%5Ccdot%29) 完成的。

有了相关系数，离注意力系数就差归一化了！其实就是用个softmax。

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200904151517126.png" alt="image-20200904151517126" style="zoom: 80%;" />

两个公式整合：

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200904151545929.png" alt="image-20200904151545929" style="zoom:67%;" />

第一步骤可以用下图表示：

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200904151617400.png" alt="image-20200904151617400" style="zoom:50%;" />

总之就是将两个特征向量进行拼接，然后经过一个前向网络学习，再softmax输出注意力系数。



(2) 加权求和(aggregate)

根据计算好的注意力系数，把特征加权求和（aggregate）一下。

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200904152219956.png" alt="image-20200904152219956" style="zoom:67%;" />

![[公式]](https://www.zhihu.com/equation?tex=h_i%5E%7B%27%7D) 就是GAT输出的对于每个顶点 ![[公式]](https://www.zhihu.com/equation?tex=i) 的新特征（融合了邻域信息）， ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5Cleft%28+%5Ccdot+%5Cright%29) 是激活函数。

Attention加入多头multi-head进行强化。

特别的，根据作者论文，在最后一步有两种做法，第一种是直接进行concat合并。

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200904152320207.png" alt="image-20200904152320207" style="zoom:67%;" />

第二种是进行求平均：作者更推荐这种方法。

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200905133952510.png" alt="image-20200905133952510" style="zoom:67%;" />

总的第二步可以用下图表示：

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20200904152356417.png" alt="image-20200904152356417" style="zoom:67%;" />

每个特征向量根据算出来的系数，进行加权求平均。

### 三、几点深入理解

**3.1 与GCN的联系与区别**

无独有偶，我们可以发现本质上而言：GCN与GAT都是将邻居顶点的特征聚合到中心顶点上（一种aggregate运算），利用graph上的local stationary学习新的顶点特征表达。不同的是GCN利用了拉普拉斯矩阵，GAT利用attention系数。一定程度上而言，GAT会更强，因为 顶点特征之间的相关性被更好地融入到模型中。

**3.2 为什么GAT适用于有向图？**

我认为最根本的原因是GAT的运算方式是逐顶点的运算（node-wise），这一点可从公式（1）—公式（3）中很明显地看出。每一次运算都需要循环遍历图上的所有顶点来完成。逐顶点运算意味着，摆脱了拉普利矩阵的束缚，使得有向图问题迎刃而解。

**3.3为什么GAT适用于inductive任务？**

GAT中重要的学习参数是![[公式]](https://www.zhihu.com/equation?tex=W) 与 ![[公式]](https://www.zhihu.com/equation?tex=a%28%5Ccdot%29) ，因为上述的逐顶点运算方式，这两个参数仅与1.1节阐述的顶点特征相关，与图的结构毫无关系。所以测试任务中改变图的结构，对于GAT影响并不大，只需要改变**![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BN%7D_i)**，重新计算即可。

与此相反的是，GCN是一种全图的计算方式，一次计算就更新全图的节点特征。学习的参数很大程度与图结构相关，这使得GCN在inductive任务上遇到困境。







### 参考

1. [GRAPH ATTENTION NETWORKS](https：//arxiv.org/abs/1710.10903)

 	2. [向往的GAT（图注意力模型）](https://zhuanlan.zhihu.com/p/81350196?utm_source=wechat_session)