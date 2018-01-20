# -*- coding: utf-8 -*-
# Author: wangchengo@126.com

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score
import pylab as pl



titanic=pd.read_csv('./titanic.txt')
# print titanic.head()
# print titanic.info()
#分离数据特征与预测目标
y=titanic['survived']
X=titanic.drop(['row.names','name','survived'],axis=1)
# print X['age']


#对(age)列缺失值进行填充
X['age'].fillna(X['age'].mean(),inplace=True)
#其余维度（非数值型）的缺失值均用unknown进行填充
X.fillna('UNKNOWN',inplace=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,
                                               random_state=33)
#对类别维度的特征进行向量化
vec=DictVectorizer()
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

# print len(vec.feature_names_)

dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
# print dt.score(X_test,y_test)



#注意  要先特征向量化之后才能进行特征选择
percentiles=np.array(range(1,100,2),dtype=int)
results=[]

for i in percentiles:
    fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    X_train_fs=fs.fit_transform(X_train,y_train)
    scores=cross_val_score(dt,X_train_fs,y_train,cv=5)
    results=np.append(results,scores.mean())
    # print X_train_fs.shape
print results

opt=np.where(results==results.max())
print 'optimanl number of features ',percentiles[opt]
opt=np.where(results==results.max())[0]


pl.plot(percentiles,results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

#只选取前7%的特征
fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=7)
X_train_fs=fs.fit_transform(X_train,y_train)
X_test_fs=fs.transform(X_test)

# print X_train_fs.shape
# print X_train.shape

dt.fit(X_train_fs,y_train)
print dt.score(X_test_fs,y_test)