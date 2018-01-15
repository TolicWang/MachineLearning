import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
column_name=['Sample code number','Clump Thickness','Uniformity of Cell Size',
              'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
              'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data=pd.read_csv('./DataSets/breast-cancer-wisconsin.data',names=column_name)



data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')

X_train,X_test,y_train,y_test=\
    train_test_split(data[column_name[1:10]],data[column_name[10]],test_size=0.25,
                     random_state=33)

#print y_train.value_counts()
#print y_test.value_counts()

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)



lr=LogisticRegression()
lr.fit(X_train,y_train)
lr_pre=lr.predict(X_test)
print lr.score(X_test,y_test)
print classification_report(y_test,lr_pre,
                            target_names=['Benign','Malignant'])



print

sgdc=SGDClassifier(loss='log')

sgdc.fit(X_train,y_train)
sgdc_pre = sgdc.predict(X_test)

print sgdc.score(X_test,y_test)

print classification_report(y_test,sgdc_pre,
                            target_names=['Benign','Malignant'])
