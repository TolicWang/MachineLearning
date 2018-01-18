#python2.7 sklearn version 0.18.1
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report
digts=load_digits()
# print digts.data.shape

X_train,X_test,y_train,y_test = train_test_split(digts.data,digts.target,
                                                 test_size=0.25,random_state=33)

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

svc=svm.SVC()
svc.fit(X_train,y_train)
y_pre=svc.predict(X_test)
print 'The Accuracy of SVC is',svc.score(X_test,y_test)
print classification_report(y_test,y_pre)
