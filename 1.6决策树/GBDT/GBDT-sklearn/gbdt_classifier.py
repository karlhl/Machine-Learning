import pydotplus

import pandas as pd
import numpy as np
from six import StringIO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics, tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt

train = pd.read_csv('train_modified.csv')
# 第一列是二元分类的输出
target = 'Disbursed'
IDcol = 'ID'
# 查看Disbursed列有多少重复值 pandas.value_counts()
print(train['Disbursed'].value_counts())

x_columns = [x for x in train.columns if x not in [target,IDcol]]
X = train[x_columns]
y = train['Disbursed']

# # 固定random_state，使每次相同参数，数据集，测试集输出一样。
# gbm0 = GradientBoostingClassifier(random_state=10)
# gbm0.fit(X,y)
# y_pred = gbm0.predict(X)
# y_predprob = gbm0.predict_proba(X)[:,1]
#
# print("Accuracy:%.4g" % metrics.accuracy_score(y.values,y_pred))
# print("AUC Score(Train): %f" % metrics.roc_auc_score(y,y_predprob))

# 对学习率和迭代次数
# param_test1 = {'n_estimators':range(20,81,10)}
# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
#                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
# gsearch1.fit(X,y)
# print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)


# 对决策树最大深度和最小样本数进行调参
# param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
# gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20,
#       max_features='sqrt', subsample=0.8, random_state=10),
#    param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
# gsearch2.fit(X,y)
# print(gsearch2.cv_results_)
# print(gsearch2.best_params_)
# print(gsearch2.best_score_)

# 内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf
# param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
# gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,
#                                      max_features='sqrt', subsample=0.8, random_state=10),
#                        param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(X,y)


# 再实验
# gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60,
#                min_samples_split =1200, max_features='sqrt', subsample=0.8, random_state=10)
# gbm1.fit(X,y)
# y_pred = gbm1.predict(X)
# y_predprob = gbm1.predict_proba(X)[:,1]
# print("Accuracy:%.4g" % metrics.accuracy_score(y.values,y_pred))
# print("AUC Score(Train): %f" % metrics.roc_auc_score(y,y_predprob))

# 最大特征数max_features
# param_test4 = {'max_features':range(7,20,2)}
# gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60,
#                min_samples_split =1200, subsample=0.8, random_state=10),
#                        param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X,y)
# print(gsearch4.cv_results_)
# print(gsearch4.best_params_)
# print(gsearch4.best_score_)

# 子采样比例
# param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
# gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60,
#                min_samples_split =1200, max_features=9, random_state=10),
#                        param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
# gsearch5.fit(X,y)
# print(gsearch5.cv_results_)
# print(gsearch5.best_params_)
# print(gsearch5.best_score_)

gbm2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=7, min_samples_leaf =60,
               min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
gbm2.fit(X,y)
y_pred = gbm2.predict(X)
y_predprob = gbm2.predict_proba(X)[:,1]
print("Accuracy:%.4g" % metrics.accuracy_score(y.values,y_pred))
print("AUC Score(Train): %f" % metrics.roc_auc_score(y,y_predprob))

dot_data = StringIO()
tree.export_graphviz(gbm2.estimators_[0, 0],
                     out_file=dot_data,
                     node_ids=True,
                     filled=True,
                     rounded=True,
                     special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("gbdt.pdf")



