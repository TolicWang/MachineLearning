# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 09:21:58 2017

@author: tolic
"""

from sklearn import svm
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size = 0.3)




svc_ovo = svm.SVC(decision_function_shape = 'ovo')
svc_ovo.fit(X_train,y_train)
print 'by one vs one: '+ str(svc_ovo.score(X_test,y_test))


svc_ova = svm.SVC(decision_function_shape = 'ova')
svc_ova.fit(X_train,y_train)
print 'by one vs all: '+ str(svc_ova.score(X_test,y_test))
