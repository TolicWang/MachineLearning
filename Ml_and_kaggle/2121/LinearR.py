# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

boston=load_boston()
# print data.DESCR
X=boston.data
y=boston.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,
                                               random_state=33)

print 'The max target value is',np.max(y)
print 'The min target value is',np.min(y)
print 'The mean target value is',np.mean(y)
# 最大值与最小值之间的差距过大，要标准化

ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)

y_train=ss_y.fit_transform(y_train)
y_test=ss_y.transform(y_test)

lr=LinearRegression()
lr.fit(X_train,y_train)
lr_y_pre=lr.predict(X_test)
print 'Default measurement ',lr.score(X_test,y_test)
print 'r2_score ',r2_score(y_test,lr_y_pre)
print 'mean_square_error ',mean_squared_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(lr_y_pre))
print 'mean_absolute_error ',mean_absolute_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(lr_y_pre))


print '---------------------------------------'

sgdr=SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_pre=sgdr.predict(X_test)
print 'Default measurement ',sgdr.score(X_test,y_test)
print 'r2_score ',r2_score(y_test,sgdr_y_pre)
print 'mean_square_error ',mean_squared_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(sgdr_y_pre))
print 'mean_absolute_error ',mean_absolute_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(sgdr_y_pre))