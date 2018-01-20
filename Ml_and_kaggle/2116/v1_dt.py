from preProgress import getData
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

X_train,X_test,y_train,y_test=getData()


######################
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pre=dtc.predict(X_test)

print 'Accuracy:',dtc.score(X_test,y_test)
print classification_report(y_test,dtc_y_pre)


######################
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pre=rfc.predict(X_test)

print 'Accuracy:',rfc.score(X_test,y_test)
print classification_report(y_test,rfc_y_pre)

######################
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pre=gbc.predict(X_test)
print 'Accuracy:',gbc.score(X_test,y_test)
print classification_report(y_test,gbc_y_pre)


