# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:34:33 2017

@author: tolic
"""

import other
from sklearn import svm
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
X,y = other.loadDataSet('ex6data1.txt') # type(X)= list

y_data = np.array(y, dtype = float)
X_data = np.array(X,dtype = float)
"""
Visualize
"""
idx_0 = np.where(y_data==0)
p0 = plt.scatter(X_data[idx_0,0],X_data[idx_0,1],
                 marker='*',color='r',label='0',s=50)

idx_1 = np.where(y_data==1)
p0 = plt.scatter(X_data[idx_1,0],X_data[idx_1,1],
                 marker='o',color='b',label='1',s=50)
plt.legend(loc = 'upper right')
plt.show()




X_train,X_test,y_train,y_test = train_test_split(
        X_data,y_data,test_size = 0.3) 

kernel_type = 'linear'

C,gamma = other.parameter(X_train,y_train,X_test,y_test
                          ,kernel_type)


svc = svm.SVC(C = 300,gamma= 1,kernel = kernel_type)
svc.fit(X_train,y_train)

other.visualize(X_train,y_train,svc,kernel_type)

print svc.score(X_test,y_test)
print svc.support_vectors_


        








        
    
