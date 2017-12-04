# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 15:58:34 2017

@author: tolic
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris();
X_data = iris.data
y_data = iris.target

"""
Step 1.
"""
scaler = StandardScaler()#
scaler.fit(X_data)
X = scaler.transform(X_data)# 标准化


m =  float(X.shape[0])
n = int(X.shape[1])

"""
Step 2.
"""
sigma =  (1/m)*X.transpose().dot(X)
"""
Step 3.
"""
U,S,V = np.linalg.svd(sigma)

diagnoal = np.diag(S)
total_variance = float(np.sum(S))
k = 0
s = 0

for i in range(0,n):
    s = s + S[i]# 确定能降到的最低维数
    if s/total_variance >= 0.99:
        k = i;
        break
u = U[:,0:k+1]

"""
Step 4.
"""
x = X.dot(u)
print x



