import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

train=pd.read_csv('./Datasets/Breast-Cancer/breast-cancer-train.csv')
test=pd.read_csv('./Datasets/Breast-Cancer/breast-cancer-test.csv')

negative=train.loc[train['Type']==0][['Clump Thickness','Cell Size']]
positive=train.loc[train['Type']==1][['Clump Thickness','Cell Size']]
plt.scatter(negative['Clump Thickness'],negative['Cell Size'],\
            marker='o',s=200,c='red')
plt.scatter(positive['Clump Thickness'],positive['Cell Size'],\
            marker='x',s=150,c ='black')
# plt.show()


X_train=train[['Clump Thickness','Cell Size']]
y_train=train['Type']
X_test=test[['Clump Thickness','Cell Size']]
y_test=test['Type']

lr=LogisticRegression()
lr.fit(X_train,y_train)
print lr.score(X_test,y_test)