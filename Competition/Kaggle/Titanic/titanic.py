# -*- coding: utf-8 -*-
# Author: wangchengo@126.com

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
import pandas as pd
from sklearn.grid_search import GridSearchCV
import datetime

train = pd.read_csv('train.csv') #
test = pd.read_csv('test.csv')

selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']



X_train=train[selected_features]
X_test=test[selected_features]

y_train=train['Survived']
X_train['Embarked'].fillna('S',inplace=True)
X_test['Embarked'].fillna('S',inplace=True)

X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)


dict_vec=DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test=dict_vec.fit_transform(X_test.to_dict(orient='record'))





gbc=GradientBoostingClassifier()
xgbc=XGBClassifier()

#
# print cross_val_score(gbc,X_train,y_train,cv=5).mean()
# print cross_val_score(xgbc,X_train,y_train,cv=5).mean()

params={'max_depth':range(2,7),'n_estimators':range(100,1200,200),
        'learning_rate':[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]}
begin=datetime.datetime.now()
gs_gbc=GridSearchCV(gbc,params,n_jobs=-1,cv=5,verbose=1)
gs_gbc.fit(X_train,y_train)
print gs_gbc.best_score_
print gs_gbc.best_params_
print datetime.datetime.now()-begin
# 0.838383838384
# {'n_estimators': 900, 'learning_rate': 0.01, 'max_depth': 4}
# 0:09:58.525028
gbc_y_pre=gs_gbc.predict(X_test)
gbc_submission=pd.DataFrame({'PassengerID':test['PassengerId'],'Survived':gbc_y_pre})
gbc_submission.to_csv('./gbc_submission.csv',index=False)


# gs_xgbc=GridSearchCV(xgbc,params,n_jobs=-1,cv=5,verbose=1)
# gs_xgbc.fit(X_train,y_train)
# print gs_xgbc.best_score_
# print gs_xgbc.best_params_
# print datetime.datetime.now()-begin
# xgbc_y_pre=gs_xgbc.predict(X_test)
# xgbc_submission=pd.DataFrame({'PassengerID':test['PassengerId'],'Survived':xgbc_y_pre})
# xgbc_submission.to_csv('./xgbc_submission.csv',index=False)
# 0.835016835017
# {'n_estimators': 900, 'learning_rate': 0.005, 'max_depth': 5}
# 0:02:36.638639