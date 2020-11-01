import pydotplus

import pandas as pd
import numpy as np
from six import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import matplotlib.pylab as plt

train = pd.read_csv('train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
train['Disbursed'].value_counts()

x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

# 迭代器数量
# param_test1 = {'n_estimators':range(10,71,10)}
# gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
#                                   min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),
#                        param_grid = param_test1, scoring='roc_auc',cv=5)
# gsearch1.fit(X,y)
# print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

# 决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split
# param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
# gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,
#                                   min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
#    param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
# gsearch2.fit(X,y)
# print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)

# 再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf
# param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
# gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, max_depth=13,
#                                   max_features='sqrt' ,oob_score=True, random_state=10),
#    param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(X,y)
# print(gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_)


# 最大特征数max_features
# param_test4 = {'max_features':range(3,11,2)}
# gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
#                                   min_samples_leaf=20 ,oob_score=True, random_state=10),
#    param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X,y)
# print(gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_)


# rf2 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
#                                   min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
# rf2.fit(X,y)
# print(rf2.oob_score_)
# y_predprob = rf2.predict_proba(X)[:,1]
# print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))







