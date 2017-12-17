import textProcess as tp
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np

data,target= tp.preProcessing()
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.30)
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pre = mnb.predict((X_test))
# print y_pre
# print y_test
print 'class count',mnb.class_count_
print 'The accuracy of Naive Bayes Classifier is',mnb.score(X_test,y_test)
print classification_report(y_test,y_pre)
