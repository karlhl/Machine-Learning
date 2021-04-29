## GCN



GCN的本质目的就是用来提取拓扑图的空间特征。





GCN的优点
1)、权值共享，参数共享，从A X W AXWAXW可以看出每一个节点的参数矩阵都是W，权值共享；
2)、具有局部性Local Connectivity，也就是局部连接的，因为每次聚合的只是一阶邻居；
上述两个特征也是CNN中进行参数减少的核心思想
3)、感受野正比于卷积层层数，第一层的节点只包含与直接相邻节点有关的信息，第二层以后，每个节点还包含相邻节点的相邻节点的信息，这样的话，参与运算的信息就会变多。层数越多，感受野越大，参与运算的信息量越充分。也就是说随着卷积层的增加，从远处邻居的信息也会逐渐聚集过来。
4)、复杂度大大降低，不用再计算拉普拉斯矩阵，特征分解

GCN的不足
1)、扩展性差：由于训练时需要需要知道关于训练节点、测试节点在内的所有节点的邻接矩阵A AA，因此是transductive的，不能处理大图，然而工程实践中几乎面临的都是大图问题，因此在扩展性问题上局限很大，为了解决transductive的的问题，GraphSAGE：Inductive Representation Learning on Large Graphs 被提出；
2)、局限于浅层：GCN论文中表明，目前GCN只局限于浅层，实验中使用2层GCN效果最好，为了加深，需要使用残差连接等trick，但是即使使用了这些trick，也只能勉强保存性能不下降，并没有提高，Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning一文也针对When GCNs Fail ？这个问题进行了分析。虽然有一篇论文：DeepGCNs-Can GCNs Go as Deep as CNNs?就是解决GCN局限于浅层的这个问题的，但个人觉得并没有解决实质性的问题，这方面还有值得研究的空间。
3)、不能处理有图：理由很简单，推导过程中用到拉普拉斯矩阵的特征分解需要满足拉普拉斯矩阵是对称矩阵的条件；
————————————————
版权声明：本文为CSDN博主「不务正业的土豆」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/yyl424525/article/details/100058264