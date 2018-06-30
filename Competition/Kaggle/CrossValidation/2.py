# -*- coding: utf-8 -*-
# Author: wangchengo@126.com
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import  SVC
from sklearn.grid_search import GridSearchCV

news=fetch_20newsgroups()

X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,
                                               test_size=0.25,random_state=3)
tfidf=TfidfVectorizer(analyzer='word',stop_words='english')

X_train_tfidf=tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.transform(X_test)

parameters={'C':np.logspace(-1,1,2),'gamma':np.logspace(-2,1,3)}
svc=SVC()
gs=GridSearchCV(svc,parameters,n_jobs=-1,cv=5,verbose=1)
gs.fit(X_train_tfidf,y_train)

print gs.best_score_
print gs.best_params_

# C:[  0.1  10. ]
# gamma [  0.01         0.31622777  10.        ]
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed: 19.0min finished
# 0.898055391868
# {'C': 10.0, 'gamma': 0.31622776601683794}

svc1=SVC(C=0.1,gamma=0.01)
svc2=SVC(C=10.0,gamma=0.316)
svc1.fit(X_train_tfidf,y_train)
svc2.fit(X_train_tfidf,y_train)
print svc1.score(X_test_tfidf,y_test)
print svc2.score(X_test_tfidf,y_test)

#0.0922587486744
#0.914457405444

