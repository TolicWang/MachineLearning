import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
data=load_iris()
x=data.data
y=data.target

index=np.where(y!=1)[0]

xx=x[index,:]
yy=y[index]*20
x_train,x_test,y_train,y_test=train_test_split(xx,yy,test_size=0.3)
print(y_test)

# model=RandomForestClassifier()
model=SVC()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
