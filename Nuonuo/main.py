import pandas as pd
import numpy as np
import re
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import SGDClassifier

def Load_Original_Traindata_Testdata_Cut(dimensions):
    train_label_data = pd.read_csv('./data/train_label.txt', names=['c1'])
    y_train = np.array(train_label_data['c1'])
    y = np.bincount(y_train)
    class_label = np.where(y < 100)[0]# class_label: a class which has less than 100 samples
    word_string=''
    train_cut_words=[]
    line_num=0
    for line in open('./data/train_text.txt',encoding='utf-8'):
        if y_train[line_num] in class_label:
            line_num+=1
            continue
        line_num+=1
        line = line.strip('\n')
        # line = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", line)
        line = re.sub("[\(\)\!\%\[\]\,\。]", "", line)
        line = line.encode('utf-8').decode('utf-8-sig')
        seg_list=jieba.cut(line,cut_all=False)
        temp=" ".join(seg_list)
        train_cut_words.append(temp)
        word_string+=(" "+temp)
    all_words = word_string.split()

    test_data = pd.read_csv('./test_text.csv', names=['data', 'label'])
    test_datas = np.array(test_data['data'])
    test_cut_words=[]
    for line in test_datas:
        line = re.sub("[\_\-\\\/\#\(\) \（ \）\!\%\[\]\、\,\。\*]", "", line)
        seg_list = jieba.cut(line, cut_all=False)
        test_cut_words.append(" ".join(seg_list))
    c=Counter()
    for x in all_words:
        if len(x)>1 and x != '\r\n':
            c[x] += 1
    most_common_words=[]
    for (k,v) in c.most_common(dimensions):
        most_common_words.append(k)
    stop_words = [item for item in all_words if item not in most_common_words]
    print('Words cut succesfully!')
    return train_cut_words,test_cut_words,stop_words

def Tf_Idf_Save(train_cut_words,test_cut_words,stop_words):
    tfidf=TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=stop_words)
    x_train=tfidf.fit_transform(train_cut_words)
    x_test=tfidf.transform(test_cut_words)
    # dict_list=tfidf.get_feature_names()
    train_label_data=pd.read_csv('./data/train_label.txt',names=['c1'])
    y_train=np.array(train_label_data['c1'])
    test_data = pd.read_csv('./test_text.csv', names=['data', 'label'])
    y_test = np.array(test_data['label'])

    #-----------------save----------------

    p={'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test}
    temp=open('./data/train_and_test_data','wb')
    pickle.dump(p,temp)
    print('Tf-Idf and Save succesfully!')
def Load_Traindata_Testdata_with_Tfidf():
    p = open('./data/train_and_test_data_10000', 'rb')
    data = pickle.load(p)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    print('Load training data succesfully! shape:',x_train.shape)
    return x_train,x_test,y_train,y_test

def train_by_partial_SGB():
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf()
    # X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.3)
    # model = RandomForestClassifier(n_jobs=-1, n_estimators=100)
    # model=XGBClassifier()
    model= SGDClassifier(n_jobs=-1,max_iter=20,alpha=0.01)
    now = datetime.datetime.now()
    print("Training begin:", now)
    batch_size=50000
    for i in range(100):
        last= datetime.datetime.now()
        start=(i*batch_size)%len(y_train)
        end=min(start+batch_size,len(y_train))
        model.partial_fit(x_train[start:end],y_train[start:end],classes=y_train)
        y_pre=model.predict(x_test)
        acc=accuracy_score(y_test,y_pre)
        score=model.score(x_test,y_test)
        cost_time=datetime.datetime.now()-last
        print("%d times,  %f score,  %f acc"%(i,score,acc),cost_time," time(s)")
    # model.fit(X_train, Y_train)
    # y_pre = model.predict(X_val)
    # print(model.score(X_val, Y_val))
    # print(accuracy_score(Y_val, y_pre))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)
def train_by_RandForest():
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf()
    model = RandomForestClassifier()
    now = datetime.datetime.now()
    print("Training begin:", now)
    params = {'max_depth': list(range(2, 7)), 'n_estimators': list(range(10, 500, 10))}
    gs = GridSearchCV(model, params, n_jobs=-1, cv=3, verbose=1)
    gs.fit(x_train,y_train)
    y_pre = gs.predict(x_test)
    print(gs.score(x_test, y_test))
    print(accuracy_score(y_test, y_pre))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)


train_cut_words,test_cut_words,stop_words=Load_Original_Traindata_Testdata_Cut(dimensions=500)
Tf_Idf_Save(train_cut_words,test_cut_words,stop_words)
# train_by_RandForest()
# x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf()

