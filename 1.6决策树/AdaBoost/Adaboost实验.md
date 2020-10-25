# Adaboost

### 一、Adaboost类库

scikit-learn中的Adaboost包含两个，AdaBoostClassifier(分类)和AdaBoostRegressor（回归）两个。

AdaBoostClassifier使用了两种Adaboost分类算法：SAMME和SAMME.R

AdaBoostRegressor使用了Adaboost.R2

### 二、框架参数

1）**base_estimator：**AdaBoostClassifier和AdaBoostRegressor都有，即我们的弱分类学习器或者弱回归学习器。理论上可以选择任何一个分类或者回归学习器，不过需要支持样本权重。我们常用的一般是CART决策树或者神经网络MLP。默认是决策树，即AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier，而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor。另外有一个要注意的点是，如果我们选择的AdaBoostClassifier算法是SAMME.R，则我们的弱分类学习器还需要支持概率预测，也就是在scikit-learn中弱分类学习器对应的预测方法除了predict还需要有predict_proba。

2）**algorithm**：这个参数只有AdaBoostClassifier有。主要原因是scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R。两者的主要区别是弱学习器权重的度量，SAMME使用了和我们的原理篇里二元分类Adaboost算法的扩展，即用对样本集分类效果作为弱学习器权重，而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重。由于SAMME.R使用了概率度量的连续值，迭代一般比SAMME快，因此AdaBoostClassifier的默认算法algorithm的值也是SAMME.R。我们一般使用默认的SAMME.R就够了，但是要注意的是使用了SAMME.R， 则弱分类学习器参数base_estimator必须限制使用支持概率预测的分类器。SAMME算法则没有这个限制。

3）**loss**：这个参数只有AdaBoostRegressor有，Adaboost.R2算法需要用到。有线性‘linear’, 平方‘square’和指数 ‘exponential’三种选择, 默认是线性，一般使用线性就足够了，除非你怀疑这个参数导致拟合程度不好。这个值的意义在原理篇我们也讲到了，它对应了我们对第k个弱分类器的中第i个样本的误差的处理，即：如果是线性误差，则$eki=|yi−Gk(xi)|/Ek$；如果是平方误差，则$eki=(yi−Gk(xi))2/E^2_k$，如果是指数误差，则$eki=1−exp（−yi+Gk(xi))Ek）$，Ek为训练集上的最大误差$Ek=max|yi−Gk(xi)|i=1,2...m$

4) **n_estimators**： AdaBoostClassifier和AdaBoostRegressor都有，就是我们的弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是50。在实际调参的过程中，我们常常将n_estimators和下面介绍的参数learning_rate一起考虑。

5) **learning_rate**:  AdaBoostClassifier和AdaBoostRegressor都有，即每个弱学习器的权重缩减系数νν，在原理篇的正则化章节我们也讲到了，加上了正则化项，我们的强学习器的迭代公式为![image-20201025144954032](https://gitee.com/karlhan/picgo/raw/master/img//image-20201025144954032.png)

。νν的取值范围为0<ν≤10<ν≤1。对于同样的训练集拟合效果，较小的νν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的ν开始调参，默认是1。

### 三、弱学习器参数

　这里我们再讨论下AdaBoostClassifier和AdaBoostRegressor弱学习器参数，由于使用不同的弱学习器，则对应的弱学习器参数各不相同。这里我们仅仅讨论默认的决策树弱学习器的参数。即CART分类树DecisionTreeClassifier和CART回归树DecisionTreeRegressor。

　　　　DecisionTreeClassifier和DecisionTreeRegressor的参数基本类似，在[scikit-learn决策树算法类库使用小结](http://www.cnblogs.com/pinard/p/6056319.html)这篇文章中我们对这两个类的参数做了详细的解释。这里我们只拿出调参数时需要尤其注意的最重要几个的参数再拿出来说一遍：

　　　　1) 划分时考虑的最大特征数**max_features**: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑log2Nlog2N个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑N−−√N个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

　　　　2) 决策树最大深**max_depth**: 默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

　　　　3) 内部节点再划分所需最小样本数**min_samples_split**: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

　　　　4) 叶子节点最少样本数**min_samples_leaf**: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

　　　　5）叶子节点最小的样本权重和**min_weight_fraction_leaf**：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

　　　　6) 最大叶子节点数**max_leaf_nodes**: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。

 

转自：

[[scikit-learn Adaboost类库使用小结](https://www.cnblogs.com/pinard/p/6136914.html)]