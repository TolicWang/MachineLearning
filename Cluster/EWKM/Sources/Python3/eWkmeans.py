# -*- coding: utf-8 -*-
# Author: wangchengo@126.com
import numpy as np
import random
import math
import metrics

def InitCentroids(X, K):
    n = np.size(X, 0)
    rands_index = np.array(random.sample(range(1, n), K))
    centriod = X[rands_index, :]
    return centriod


def findClosestCentroids(X,lambdaa,centroids):
    K = np.size(centroids, 0)
    n,m=X.shape
    idx = np.zeros((np.size(X, 0)), dtype=int)
    for i in range(n):
        subs = centroids - X[i,:]# k by m
        distance2=np.power(subs, 2)# lambdaa  k by m
        w_distance2=np.multiply(lambdaa,distance2)
        w_distance2_sum=np.sum(w_distance2,axis=1)

        if math.isnan(w_distance2_sum.sum()) or math.isinf(w_distance2_sum.sum()):
            w_distance2_sum=np.zeros(K)
            # print 'the situation that w_distance2_sum is nan or inf'
        idx[i] = np.where(w_distance2_sum == w_distance2_sum.min())[0][0]
    return idx
def computeCentroids(X,idx,K):
    n,m=X.shape
    centriod = np.zeros((K, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]  # 一个簇一个簇的分开来计算
        temp = X[index, :]  # ? by m # 每次先取出一个簇中的所有样本
        s = np.sum(temp, axis=0)
        centriod[k, :] = s / np.size(index)
    return centriod
def computeWeight(X,centroids,idx,gamma):
    K = np.size(centroids,0)
    n,m=X.shape
    lambdaa=np.zeros((K,m),dtype=float)
    D = np.zeros((K, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]  # ? by m
        distance2 = np.power((temp - centroids[k,:]), 2)  # ? by m
        D[k,:]=np.sum(distance2,axis=0)
    for i in range(K):
        numerator=np.exp(np.divide(-D[i,:],gamma))
        denominator=np.sum(numerator,axis=0)
        lambdaa[i,:]=np.divide(numerator,denominator)
    return lambdaa
def costFunction(X,centroids,idx,lambdaa,gamma):
    cost = 0
    K = np.size(centroids, 0)
    n, m = X.shape
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]# ? by m
        distance2 = np.power((temp - centroids[k,:]),2)  # ? by m
        w_distance2=np.multiply(lambdaa[k,:],distance2)# 1 by m
        temp = gamma*np.dot(lambdaa[k,:],np.log(lambdaa[k,:]))
        cost = cost + np.sum(w_distance2)+temp
    return  cost
def isConvergence(costF):
    if math.isnan(np.sum(costF)):
        return False
    iteration = np.size(costF)
    for i in range(iteration-1):
        if costF[i] < costF[i+1]:
            return False
        i=i+1
    return True
def ewkmeans(X,K,gamma,iterations):
    n, m = X.shape
    costF = np.zeros(iterations)
    lambdaa = np.zeros((K, m), dtype=float) + np.divide(1, float(m))
    centroids = InitCentroids(X, K)
    for i in range(iterations):
        idx = findClosestCentroids(X,lambdaa,centroids)
        centroids = computeCentroids(X,idx,K)
        lambdaa =computeWeight(X,centroids,idx,gamma)
        costF[i] = costFunction(X,centroids,idx,lambdaa,gamma)
    best_labels = idx
    best_centers=centroids
    if not (math.isnan(np.sum(costF))) and isConvergence(costF):
        isConverge=True
        return isConverge,best_labels,best_centers
    else:
        isConverge=False
        return isConverge,None,None

class EWKmeans:
    n_cluster=0
    max_iter=0
    gamma=0
    best_labels, best_centers = None, None
    isConverge=False

    def __init__(self,n_cluster=3,max_iter=20,gamma=10.0):
        self.n_cluster=n_cluster
        self.max_iter=max_iter
        self.gamma=gamma


    def fit(self,X):
        self.isConverge,self.best_labels,self.best_centers=ewkmeans(
            X=X,K=self.n_cluster,gamma=self.gamma,iterations=self.max_iter
        )
        return self

    def fit_predict(self,X,y=None):
        if self.fit(X).isConverge:
            return self.best_labels
        else:
            return 'Not convergence with current parameter ' \
                   'or centroids,Please try again'
    def get_params(self):
        return self.isConverge,self.n_clusters,self.gamma

