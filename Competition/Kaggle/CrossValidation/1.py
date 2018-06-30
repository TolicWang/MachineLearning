# -*- coding: utf-8 -*-
# Author: wangchengo@126.com

from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC

datas=load_iris()
X=datas.data
y=datas.target



rbf_c1=SVC(kernel='rbf',C=1.0)
rbf_c2=SVC(kernel='rbf',C=2.0)

result1=cross_val_score(rbf_c1,X,y,cv=6)
print result1
print result1.mean()

result2=cross_val_score(rbf_c2,X,y,cv=6)
print result2
print result2.mean()
