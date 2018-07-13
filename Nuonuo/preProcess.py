# -*- coding: utf-8 -*-
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import  train_test_split
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
def load_data_and_cut(train_dir,label_dir):
    data=pd.read_csv(label_dir,names=['c1'])
    labels=np.array(data['c1'])
    word_string=''
    content=[]
    for line in open(train_dir,encoding='utf-8'):
        line = line.strip('\n')
        # line = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", line)
        line = re.sub("[\(\)\!\%\[\]\,\。]", "", line)
        line = line.encode('utf-8').decode('utf-8-sig')
        seg_list=jieba.cut(line,cut_all=False)
        temp=" ".join(seg_list)
        content.append(temp)
        word_string+=(" "+temp)
    c=Counter()
    all_words=word_string.split()
    for x in all_words:
        if len(x)>1 and x != '\r\n':
            c[x] += 1
    print('常用词频度统计结果')
    most_common_words=[]
    for (k,v) in c.most_common(10):
        most_common_words.append(k)
        # print('%s:%d' % (k, v))
    # print("most_common_words",most_common_words)
    # print("all_words",all_words)
    stop_words=[item for item in all_words if item not in most_common_words]
    # print("stop_words",stop_words)
    # print("word vector",content)
    return content,labels,stop_words
def tf_idf(train_data,test_data,stop_words):
    tfidf=TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=stop_words)
    x_train=tfidf.fit_transform(train_data)
    x_test=tfidf.transform(test_data)
    dict_list=tfidf.get_feature_names()
    return x_train,x_test,dict_list

def get_batch(X,labels,batch_size):
    test_size=batch_size/len(labels)
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=test_size)
    return x_test,y_test


def train_and_test():

    #training---------------------------------------
    label_dir='./train_label.txt'
    train_dir="./train_text.txt"
    # train_dir="./batch.txt"
    train_data,train_labels,stop_words=load_data_and_cut(train_dir,label_dir)
    # print("stop_words", stop_words)
    # print("cut result",content)

    testData=pd.read_csv('./test_text.csv',names=['data','label'])
    test_data=np.array(testData['data'])
    test_label=np.array(testData['label'])
    X_train,X_test,dict_list=tf_idf(train_data,test_data,stop_words)

    print(train_data)
    print(test_data)
    # batch_x,batch_y=get_batch(X_train,train_labels,5000)
    #
    # params={'max_depth':list(range(2,7)),'n_estimators':list(range(10,2000,100))}
    # forest=RandomForestClassifier()
    # gs=GridSearchCV(forest,params,n_jobs=-1,cv=5,verbose=1)
    # gs.fit(batch_x,batch_y)
    # y_pre=gs.predict(X_test)
    # print(gs.score(X_test,test_label))
    # print(accuracy_score(test_label,y_pre))

train_and_test()

#
#
#
#

#
# model=LogisticRegression()
#
# percentiles=np.array(range(1,10,2),dtype=int)
# results=[]
# for i in percentiles:
#     fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i) # percentile表示选取前%i 的特征
#     x_train_fs=fs.fit_transform(x_test,y_test)
#     scores=cross_val_score(model,x_train_fs,y_train,cv=1) #5折交叉验证，返回5次验证后的scores
#     results=np.append(results,scores.mean()) #得到每次取的前%i特征所产生的score的均值
#     print(x_train_fs.shape)#可以查看每次的新维度
# print(results)
#
# opt=np.where(results==results.max())[0]
# print ('\noptimanl number of features ',percentiles[opt])