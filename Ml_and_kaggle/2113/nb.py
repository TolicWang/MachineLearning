#python2.7 sklearn version 0.18.1
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news=fetch_20newsgroups()
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,
                                               test_size=0.25,random_state=33)


countVec=CountVectorizer()
count_train = countVec.fit_transform(X_train)
count_test = countVec.transform(X_test)

mnb=MultinomialNB()
mnb.fit(count_train,y_train)
mnb_pre=mnb.predict(count_test)

print 'Accuracy:',mnb.score(count_test,y_test)
print classification_report(y_test,mnb_pre,target_names=news.target_names)
