# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 19:54:04 2017

@author: tolic
"""
import numpy as np

def loadDataSet(fileNmae):
    dataMat = []
    labelMat = []
    fr = open(fileNmae)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def sigmoid(z):
    gz = 1.0/(1+np.exp(-z))
    return gz

def gradDescent(X, y, W, b, alpha):
    maxIteration = 500
    for i in range(maxIteration):
        z = X*W+b
        error = sigmoid(z)- y
        W = W - (1.0/m)*alpha*X.T*error
        b = b - (1.0/m)*alpha*np.sum(error)
    return W,b

def accuracy(X,y,W,b):
    m,n = np.shape(X)
    prob = sigmoid(X*W+b)
    predictioin = np.ones((m,1),dtype = float)
    for i in range(m):
        if prob[i,0] < 0.5:
            predictioin[i] = 0.0
    return 1-np.sum(np.abs(y - predictioin))/m
    

X,y = loadDataSet('data.txt')
m,n = np.shape(X)
X = np.mat(X)
y = np.mat(y).T



W = -5*np.ones((n,1),dtype = float)

b = -5.1
alpha = 0.0013

W,b = gradDescent(X, y, W, b, alpha)
print "******************"
print "W is :             " 
print W
print "accuracy is :         " + str(accuracy(X, y, W, b))
print "******************"









    