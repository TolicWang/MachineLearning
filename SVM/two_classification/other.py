# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:56:46 2017

@author: tolic
"""
import random 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def loadDataSet(fileNmae):
    dataMat = []
    labelMat = []
    fr = open(fileNmae)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def visualize(X,y,svc,kernel):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
     np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with '+kernel+' kernel')
    plt.show()

def parameter(X_train,y_train,X_test,y_test,kernel_type):
    c = np.array([0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000])
    g = np.array([0.01,0.03,1,3,10,30,100])
    max_score = 0
    for i in range(0,c.size):
        for j in range(0,g.size):
            svc = svm.SVC(C=c[i],gamma=g[j],kernel = kernel_type)
            svc.fit(X_train,y_train)
            s = svc.score(X_test,y_test) 
            if (s > max_score):           
                C = c[i]
                gamma = g[j]
                max_score = s
    return C,gamma        
             
             
             
             
             
             
             
             
             
             
             
             
            