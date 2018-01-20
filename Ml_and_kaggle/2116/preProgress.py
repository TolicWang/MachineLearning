import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

def getData():
    titanic=pd.read_csv('./titanic.txt')
    X=titanic[['pclass','age','sex']]
    y=titanic['survived']
    X['age'].fillna(X['age'].mean(),inplace=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,
                                                   random_state=33)
    vec=DictVectorizer(sparse=False)
    X_train=vec.fit_transform(X_train.to_dict(orient='record'))
    X_test=vec.transform(X_test.to_dict(orient='record'))
    return X_train,X_test,y_train,y_test



