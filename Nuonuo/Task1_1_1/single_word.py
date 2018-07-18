import datetime
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from sklearn.linear_model import SGDClassifier
def check_contain_chinese(check_str):
     for ch in check_str.decode('utf-8'):
         if u'\u4e00' <= ch <= u'\u9fff':
             return True
     return False

def Load_Original_Traindata_Testdata_Cut_and_Save():
    now=datetime.datetime.now()
    f=open('./data/train_cut_words.txt','w',encoding='utf-8')
    for line in open('../data/train_text.txt',encoding='utf-8'):
        line = line.strip('\n')
        line = re.sub("[0-9\_\-\\\/\#\(\) \（ \）\!\%\[\]\、\,\。\*\+\【\】\.\”\“ \: ]", "", line)
        line = line.encode('utf-8').decode('utf-8-sig')
        temp = list(line)
        s = ""
        for i in range(len(temp)):
            s += (" " + temp[i])
        f.write(s+'\n')
    f.close()

    test_data = pd.read_csv('../data/test_text.csv', names=['data', 'label'])
    test_datas = np.array(test_data['data'])
    f=open('./data/test_cut_words.txt','w',encoding='utf-8')
    for line in test_datas:
        line = re.sub("[0-9\_\-\\\/\#\(\) \（ \）\!\%\[\]\、\,\。\*\+\【\】\.\”\“ \: ]", "", line)
        temp = list(line)
        s = ""
        for i in range(len(temp)):
            s += (" " + temp[i])
        f.write(s + '\n')
    f.close()
    last=datetime.datetime.now()
    print('Train and Test words Cut and Save succesfully!',last-now," time(s)")
def Make_Train_and_Test_Tf_Idf_and_Save():
    now = datetime.datetime.now()
    print("Make Train and Test Tf-Idf begin: ", now)
    train_cut_words=[]
    for line in open('./data/train_cut_words.txt', encoding='utf-8'):
        line=line.strip('\n')
        train_cut_words.append(line)
    test_cut_words=[]
    for line in open('./data/test_cut_words.txt', encoding='utf-8'):
        line=line.strip('\n')
        test_cut_words.append(line)
    tfidf=TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    x_train=tfidf.fit_transform(train_cut_words)
    x_test=tfidf.transform(test_cut_words)

    train_label_data=pd.read_csv('../data/train_label.txt',names=['c1'])
    y_train=np.array(train_label_data['c1'])
    test_data = pd.read_csv('../data/test_text.csv', names=['data', 'label'])
    y_test = np.array(test_data['label'])

    #-----------------save----------------

    p={'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test}
    temp=open('./data/train_and_test_data','wb')
    pickle.dump(p,temp)
    last=datetime.datetime.now()
    print('Make Train and Test Tf-Idf with dimension:',x_train.shape[1]
          ,' and Save succesfully!',last-now,' time(s)')
def Load_Traindata_Testdata_with_Tfidf(filename):
    p = open('./data/'+filename, 'rb')
    data = pickle.load(p)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    print('Load training data succesfully! shape:',x_train.shape)
    return x_train,x_test,y_train,y_test
def Feature_Section_by_RandForest():
    filename = 'train_and_test_data'
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf(filename)
    model = RandomForestClassifier(n_jobs=8,n_estimators=15)
    now = datetime.datetime.now()
    print("Feature sele ction Training begin by RandForest:", now)
    model.fit(x_train,y_train)
    y_pre = model.predict(x_test)
    print(model.score(x_test, y_test))
    print(accuracy_score(y_test, y_pre))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)
    now = datetime.datetime.now()
    print("Feature selection:",now)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    p={'indices':indices }
    temp=open('./data/indices','wb')
    pickle.dump(p,temp)
    last=datetime.datetime.now()
    print('Feature importance indices get succesfully!',last-now,' time(s)')

def train_by_RandForest():
    filename='train_and_test_data'
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf(filename)
    p = open('./data/indices', 'rb')
    data = pickle.load(p)
    indices = data['indices']
    most_importance_feature = indices[:2000]
    x_train = x_train[:,most_importance_feature]
    x_test = x_test[:,most_importance_feature]
    print("Selected feature with shape:",x_train.shape)
    model = RandomForestClassifier(n_jobs=8,n_estimators=30)
    now = datetime.datetime.now()
    print("Training begin by RandForest:", now)
    model.fit(x_train,y_train)
    y_pre = model.predict(x_test)
    print(model.score(x_test, y_test))
    print(accuracy_score(y_test, y_pre))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)

Load_Original_Traindata_Testdata_Cut_and_Save()
Make_Train_and_Test_Tf_Idf_and_Save()
Feature_Section_by_RandForest()
train_by_RandForest()
