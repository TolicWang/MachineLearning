from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
data=load_digits()
X=data.data
y=data.target*2+5
# y=data.target
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=22)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
lr=LogisticRegression()
model=lr
model.fit(X_test,y_test)
print(model.score(X_test,y_test))
#


