import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

train=pd.read_csv('./Datasets/Breast-Cancer/breast-cancer-train.csv')
test=pd.read_csv('./Datasets/Breast-Cancer/breast-cancer-test.csv')

train_data = np.array(train)
test_data = np.array(test)

X_train = train_data[:,1:3]
y_train = train_data[:,3]

X_test = test_data[:,1:3]
y_test = test_data[:,3]


p_index = np.where(train_data[:,3]==1)[0]
n_index = np.where(train_data[:,3]==0)[0]
positive = X_train[p_index,:]
nagative = X_train[n_index,:]

plt.scatter(nagative[:,0],nagative[:,1],marker='o',s=200,c='red')
plt.scatter(positive[:,0],positive[:,1],marker='x',s=150,c='black')
plt.show()


lr=LogisticRegression()
lr.fit(X_train,y_train)
print lr.score(X_test,y_test)


