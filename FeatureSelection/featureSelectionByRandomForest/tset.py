from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data=load_wine()
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.3,random_state=22)
forest=RandomForestClassifier(n_estimators=10,n_jobs=-1,random_state=9)
forest.fit(x_train,y_train)
print(forest.score(x_test,y_test))
importances=forest.feature_importances_
indices = np.argsort(importances)[::-1]# a[::-1]让a逆序输出
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, data.feature_names[f], importances[indices[f]]))

