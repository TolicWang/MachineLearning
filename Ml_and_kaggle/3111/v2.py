# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news=fetch_20newsgroups(subset='all')
# print news.data[0]
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,
                                               test_size=0.25,random_state=33)

#########   Naive Bayes with CountVectorizer

count_vec=CountVectorizer(stop_words='english')
X_cout_train=count_vec.fit_transform(X_train)
X_cout_test=count_vec.transform(X_test)

mnb=MultinomialNB()
mnb.fit(X_cout_train,y_train)

print 'The accuracy of classifying 20news using Naive Bayes(countvectorizer with ' \
      'filtering stopwords):',mnb.score(X_cout_test,y_test)

y_count_pre=mnb.predict(X_cout_test)
print classification_report(y_test,y_count_pre,target_names=news.target_names)

#########

