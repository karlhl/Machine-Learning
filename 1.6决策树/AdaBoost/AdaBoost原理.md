# AdaBoost

是集成学习中Boosting中最典型的代表。



### 基本原理

Adaboost算法基本原理就是将多个弱分类器（弱分类器一般选用单层决策树）进行合理的结合，使其成为一个强分类器。

Adaboost采用迭代的思想，每次迭代只训练一个弱分类器，训练好的弱分类器将参与下一次迭代的使用。也就是说，在第N次迭代中，一共就有N个弱分类器，其中N-1个是以前训练好的，其各种参数都不再改变，本次训练第N个分类器。其中弱分类器的关系是第N个弱分类器更可能分对前N-1个弱分类器没分对的数据，最终分类输出要看这N个分类器的综合效果。

##### 采用了弱分类器——单层决策树

单层决策树是决策树的最简化版本，只有一个决策点，在单层决策树中，一共只有一个决策点，所以只能在其中一个维度中选择一个合适的决策阈值作为决策点。

Adaboost包含两种权重，一种是数据的权重:用于弱分类器寻找其分类误差最小的决策点，然后计算出该分类器的权重，也就是第二种权重。分类器权重代表在最终决策时有更大的发言权。

数据的权重：在Adaboost算法中，每训练完一个弱分类器都就会调整权重，上一轮训练中被误分类的点的权重会增加，在本轮训练中，由于权重影响，本轮的弱分类器将更有可能把上一轮的误分类点分对，如果还是没有分对，那么分错的点的权重将继续增加，下一个弱分类器将更加关注这个点，尽量将其分对。权重大的点得到更多的关注，权重小的点得到更少的关注。

分类器的权重：由于Adaboost中若干个分类器的关系是第N个分类器更可能分对第N-1个分类器没分对的数据，而不能保证以前分对的数据也能同时分对。所以在Adaboost中，每个弱分类器都有各自最关注的点，每个弱分类器都只关注整个数据集的中一部分数据，所以它们必然是共同组合在一起才能发挥出作用。所以最终投票表决时，需要根据弱分类器的权重来进行加权投票，权重大小是根据弱分类器的分类错误率计算得出的，总的规律就是弱分类器错误率越低，其权重就越高。

![Adaboost分类器结构](https://gitee.com/karlhan/picgo/raw/master/img//aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwNjA5MTY0ODI1NDM4)



如图所示为Adaboost分类器的整体结构。从右到左，可见最终的求和与符号函数，再看到左边求和之前，图中的虚线表示不同轮次的迭代效果，第1次迭代时，只有第1行的结构，第2次迭代时，包括第1行与第2行的结构，每次迭代增加一行结构，图下方的“云”表示不断迭代结构的省略。

第i轮迭代要做这么几件事：

1. 新增弱分类器WeakClassifier(i)与弱分类器权重alpha(i)
2. 通过数据集data与数据权重W(i)训练弱分类器WeakClassifier(i)，并得出其分类错误率，以此计算出其弱分类器权重alpha(i)
3. 通过加权投票表决的方法，让所有弱分类器进行加权投票表决的方法得到最终预测输出，计算最终分类错误率，如果最终错误率低于设定阈值（比如5%），那么迭代结束；如果最终错误率高于设定阈值，那么更新数据权重得到W(i+1)

##### 实验例子

![这里写图片描述](https://gitee.com/karlhan/picgo/raw/master/img//aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwNjA5MTcwMTU1NjQ3)

Adaboost分类器试图把两类数据分开，运行一下程序，显示出决策点，如下图：

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwNjA5MTcwMzM4MTAz?x-oss-process=image/format,png)

这样一看，似乎是分开了，不过具体参数是怎样呢？查看程序的输出，可以得到如其决策点与弱分类器权重，在图中标记出来如下：

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwNjA5MTcwNTA2NTQz?x-oss-process=image/format,png)

图中被分成了6分区域，每个区域对应的类别就是：
1号：sign(-0.998277+0.874600-0.608198)=-1
2号：sign(+0.998277+0.874600-0.608198)=+1
3号：sign(+0.998277+0.874600+0.608198)=+1
4号：sign(-0.998277-0.874600-0.608198)=-1
5号：sign(+0.998277-0.874600-0.608198)=-1
6号：sign(+0.998277-0.874600+0.608198)=+1
其中sign(x)是符号函数，正数返回1负数返回-1。
最终得到如下效果：
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwNjA5MTcwNzAzNzgw?x-oss-process=image/format,png)



### sklearn实现

```python
from sklearn.ensemble import AdaBoostClassifier # For Classification
from sklearn.ensemble import AdaBoostRegressor  # For Regression
from skleran.tree import DecisionTreeClassifier
 
dt = DecisionTreeClassifier()
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt, learning_rate=1)
# Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it accepts sample weight
clf.fit(x_train, y_train)
```

可以调整参数以优化算法的性能：

- n_estimators：它控制了弱学习器的数量
- learning_rate：控制在最后的组合中每个弱分类器的权重，需要在learning_rate和n_estimators间有个权衡
- base_estimators：它用来指定不同的ML算法。

也可以调整基础学习器的参数以优化它自身的性能。



参考：

- [Adaboost入门教程](https://blog.csdn.net/px_528/article/details/72963977)























