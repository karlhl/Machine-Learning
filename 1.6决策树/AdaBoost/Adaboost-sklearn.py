"""
利用sklearn生成分类数据，然后用adaboot进行分类
"""

from sklearn.ensemble import AdaBoostClassifier  # For Classification
from sklearn.ensemble import AdaBoostRegressor  # For Regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib.pyplot as plt

# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1, y1 = make_gaussian_quantiles(cov=2.0,n_samples=500, n_features=2,n_classes=2, random_state=1)
# 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=400, n_features=2, n_classes=2, random_state=1)
#讲两组数据合成一组数据
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show()

# 用bdt进行分类拟合
# n_estimators(弱分类器数量)，algorithm(选用的算法)
# 弱分类器越多，拟合程度越好，但容易过拟合
# 同样的弱分类器的个数情况下，如果减少步长，拟合效果会下降
# base_estimator是学习期，这里是DecisionTreeClassifier,是CART回归树
dt = DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5)
bdt = AdaBoostClassifier(base_estimator=dt,
                         algorithm="SAMME.R",
                         n_estimators=300, learning_rate=0.8)
bdt.fit(X, y)
"""
AdaBoostClassifier(algorithm='SAMME',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=5,
            min_samples_split=20, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'),
          learning_rate=0.8, n_estimators=200, random_state=None)
"""

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

# 拟合分数
print("score:",bdt.score(X,y))
plt.show()











# dt = DecisionTreeClassifier()
# clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt, learning_rate=1)
# # Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it accepts sample weight
# clf.fit(x_train, y_train)