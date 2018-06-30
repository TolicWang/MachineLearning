# -*- coding: utf-8 -*-
# Author: wangchengo@126.com
from eWkmeans import EWKmeans
from metrics import Metrics
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
from DataSet.preProcessing import load_DataSets


'''
NOTE 1:
本算法是仿造sklearn中的设计模式实现的，用法示例如下（test.py)
但需要注意的是
fit_predict()方法的返回值分两种情况：  
(1) 如果收敛，则返回标签；
(2) 如果发散，则返回'Not convergence with current parameter ' \
                   'or centroids,Please try again '
NOTE 2:
由于只是简单的实现算法，在编程的时候并没有考虑跟多的细节
所以： 
(1).当参数设置不太合理的时候，需要多运行几次算法才会收敛，此时可以借用下面6行代码
(2).极端情况下（例如eta过大），算法几乎不会收敛

NOTE 3:
(1).Metrics(y_true,y_pre) 是聚类中常用的4种评价指标的合集；
(2).getFscAccNmiAri() 依次返回的是Fsc,Acc,NMI,ARI
'''

X,y,K=load_DataSets('Astrilian_0',isScaler=True)
ewk=EWKmeans(n_cluster=3,gamma=10.0)

y1_pre=ewk.fit_predict(X)
if ewk.isConverge:
    print('Result of WKmeans:', Metrics(y, y1_pre).getFscAccNmiAri())
else:
    print(y1_pre)
#
# count = 1
# while not ewk.isConverge:
#     y1_pre = ewk.fit_predict(X)
#     print('Runing times:',count)
#     count+=1
# print('Result of EWk:', Metrics(y, y1_pre).getFscAccNmiAri())





