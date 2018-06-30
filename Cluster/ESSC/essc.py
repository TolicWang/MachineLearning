# -*- coding: utf-8 -*-
# Author: wangchengo@126.com
import numpy as np
import random
import math
import metrics

def InitVAndV0(X,K): # V namely the centroids
    N, D = X.shape  # sampe Number and sample Dimension
    rands_index = np.array(random.sample(range(1, N), K))
    V=X[rands_index, :]
    V0 = np.sum(X,axis=0)/float(N)
    return V,V0

def findU(X,w,V,V0,m,eta):# V0 is the global center
    N,D=X.shape # sampe Number and sample Dimension
    K = np.size(V, 0)
    d = np.zeros((K, N), dtype=float)
    term1=np.zeros((K,N),dtype=float)
    u=np.zeros((K,N),dtype=float)
    for k in range(K):
        # compute the first term:
        temp1=w[k,:]*np.power((X-V[k,:]),2)# 1 by D  *  N by D ==> N by D
        temp1=np.sum(temp1,axis=1)# N by 1  每一行相加
        term1[k,:]=temp1# 1 by N
        # compute the second term:
        temp2=w[k,:]*np.power((V[k,:]-V0),2)# 1 by D * 1 by D ==> 1 by D
        temp2=temp2.reshape((1,D))
        temp2=eta*np.sum(temp2)# 1 by 1
        d[k,:] = term1[k,:]-temp2# 1 by N
        exponent = (-1) / float(m - 1)
        s=np.power(d,exponent)
        s=np.nan_to_num(s)
    for k in range(K):
        #numerator
        numerator =np.power(d[k,:],exponent)# 1 by N
        numerator = np.nan_to_num(numerator)
        numerator =numerator.reshape(1,N)
        #denominator
        denominator = np.sum(s,axis=0)  # 1 by N
        denominator = denominator.reshape(1,N)
        u[k,:]=numerator/(denominator)


    return u

def computeV(X,u,eta,m,K,V0):
    N, D = X.shape  # sampe Number and sample Dimension
    v=np.zeros((K,D),dtype=float)
    for k in range(K):
        p=np.power(u[k,:],m)
        p=p.reshape(1,N)
        numerator=np.sum(np.dot(p,(X-eta*V0)),axis=0) # 1 by D
        denominator=np.sum(np.power(u[k,:],m))*(1-eta)# 1 by 1
        v[k, :] = numerator/float(denominator)
    return v

def computeWeight(X,V,V0,m,u,eta,gamma):
    N, D = X.shape  # sampe Number and sample Dimension
    K = np.size(V, 0)
    w=np.zeros((K,D),dtype=float)
    sigma=np.zeros((K,D),dtype=float)
    for k in range(K):
        #first term
        temp1 = np.power(u[k,:],m) # 1 by N
        temp2 = np.power((X-V[k,:]),2) # N by D
        term1 = np.dot(temp1,temp2)# 1 by D
        #second term
        temp3=np.power((V[k,:]-V0),2) # 1 by D
        term2=np.zeros((1,D),dtype=float)
        for i in range(temp1.size):
            term2=term2+temp1[i]*temp3
        sigma[k,:]=term1-eta*term2

    for k in range(K):
        numerator=np.exp(-sigma[k,:]/gamma)
        w[k,:] = numerator/np.sum(numerator)


    return w

def costFunction(X,u,V,V0,eta,gamma,w,m):
    cost=0
    #J1-----------------------------
    J1=0
    N, D = X.shape  # sampe Number and sample Dimension
    K = np.size(V, 0)
    for k in range(K):
        p=np.power((X-V[k,:]),2)# N by D
        mul0=w[k,:]*p#  1 by D  *   N by D ==> N by D
        u_m1=np.power(u[k,:],m) # 1 by N
        J1=J1+np.sum(np.dot(u_m1,mul0))

    # J2-----------------------------
    J2=0
    J2=gamma*np.sum(w*np.log(w))
    J2=np.nan_to_num(J2)

    # J3-----------------------------
    J3=0
    for k in range(K):
        mul3=w[k,:]*np.power((V[k,:]-V0),2) # 1 by D
        u_m3=np.power(u[k,:],m)
        u_m3=np.nan_to_num(u_m3)

        J3=J3+eta*np.sum(u_m3)*np.sum(mul3)


    cost = J1+J2-J3
    return cost

def isConvergence(costF,max_iter):
    if math.isnan(np.sum(costF)):
        return False
    index = np.size(costF)
    for i in range(index-1):
        if costF[i] < costF[i+1]:
            return False
    if index >= max_iter:
        return True
    elif costF[index-1]==costF[index-2]==costF[index-3]:
        return True
    return 'continue'

def essc(X,K,gamma,eta,max_iter):
    N, D = X.shape
    if (min(N, D - 1)) >= 3:
        m = min((N, D - 1)) / float((min(N, D - 1) - 2))
    else:
        m = 2
    costF = []
    V, V0 = InitVAndV0(X, K)
    w = np.random.rand(K, D)
    s = np.sum(w, axis=1).reshape((K, 1))
    w = w / s
    for i in range(max_iter):
        u=findU(X,w,V,V0,m,eta)
        V=computeV(X,u,eta,m,K,V0)
        w=computeWeight(X,V,V0,m,u,eta,gamma)
        c=costFunction(X,u,V,V0,eta,gamma,w,m)
        costF.append(round(c,4))

        if i < 2 :
            continue
        flag=isConvergence(costF,max_iter)
        if flag=='continue':
            continue
        elif flag:
            u = np.nan_to_num(u)
            max = np.max(u, axis=0)
            idx = np.zeros((N), dtype=int)
            for i, ma in enumerate(max):
                idx[i] = np.where(u == ma)[0][0]
            best_labels = idx
            best_centers = V
            isConverge = True
            return isConverge, best_labels, best_centers, costF

        else:
            isConverge=False
            return isConverge,None,None,costF

class ESSC:
    n_clusters = 0
    max_iter = 0
    gamma = 0
    best_labels, best_centers = None, None
    eta = 0
    isConverge = False
    cost=None
    def __init__(self,n_clusters=3,max_iter=100,gamma=10.0,eta=0.001):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.gamma=gamma
        self.eta=eta
    def fit(self,X,y=None):
        self.isConverge,self.best_labels,self.best_centers,self.cost=essc(
            X=X,K=self.n_clusters,gamma=self.gamma,eta=self.eta,
            max_iter=self.max_iter)
        return self
    def fit_predict(self,X,y=None):

        if self.fit(X).isConverge:
            return self.best_labels
        else:
            return 'Not convergence with current parameter ' \
                   'or centroids,Please try again'
    def get_params(self):
        return self.isConverge,self.n_clusters,self.gamma,self.eta
    def get_cost(self):
        return self.cost

