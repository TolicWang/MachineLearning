import pandas as pd
import numpy as np
import re
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import  KMeans

def Load_Original_Traindata_Testdata_Cut_and_Save():
    now=datetime.datetime.now()
    print("Train and Test words Cut begin: ",now)
    train_label_data = pd.read_csv('./data/train_label.txt', names=['c1'])
    y_train = np.array(train_label_data['c1'])
    y_count = np.bincount(y_train)
    less_than_label = np.where(y_count <1)[0]
    # print(less_than_label)
    # remove samples which appear in a class less than 5 times
    line_number = 0
    count=0
    f=open('./data/train_cut_words.txt','w',encoding='utf-8')
    p=open('./data/train_label_remove_less_than.txt','w',encoding='utf-8')
    for line in open('./data/train_text.txt',encoding='utf-8'):
        if y_train[line_number] in less_than_label:
            line_number += 1
            continue
        p.write(str(y_train[line_number])+'\n')
        line_number += 1
        count+=1
        line = line.strip('\n')
        # line = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", line)
        line = re.sub("[\_\-\\\/\#\(\) \（ \）\!\%\[\]\、\,\。\*\+\【\】]", "", line)
        line = line.encode('utf-8').decode('utf-8-sig')
        seg_list=jieba.cut(line,cut_all=False)
        temp=" ".join(seg_list)
        f.write(temp+'\n')

    f.close()
    p.close()
    test_data = pd.read_csv('./data/test_text.csv', names=['data', 'label'])
    test_datas = np.array(test_data['data'])
    f=open('./data/test_cut_words.txt','w',encoding='utf-8')
    for line in test_datas:
        line = re.sub("[\_\-\\\/\#\(\) \（ \）\!\%\[\]\、\,\。\*\+\【\】]", "", line)
        seg_list = jieba.cut(line, cut_all=False)
        temp=" ".join(seg_list)
        f.write(temp+'\n')
    f.close()
    last=datetime.datetime.now()
    print('Train and Test words Cut and Save succesfully!',last-now," time(s)")
def Make_Stop_Words_and_Save(dimensions):
    now=datetime.datetime.now()
    print("Make stop words begin: ",now)
    all_candidate_stop_words = ""
    for line in open('./data/train_cut_words.txt', encoding='utf-8'):
        line = line.strip('\n')+" "
        all_candidate_stop_words += line
    all_candidate_stop_words=all_candidate_stop_words.split()
    c=Counter()
    for x in all_candidate_stop_words:
        if len(x)>1 and x != '\r\n':
            c[x] += 1
    most_common_words=[]
    # print('all_candidate_stop_words len:',len(all_candidate_stop_words))
    for (k,v) in c.most_common(dimensions):
        most_common_words.append(k)
    # print('most_common_words len:',len(most_common_words))
    stop_words = [item for item in all_candidate_stop_words if item not in most_common_words]
    stop_words = list(set(stop_words))
    p={'stop_words':stop_words}
    temp=open('./data/stop_words','wb')
    pickle.dump(p,temp)
    last=datetime.datetime.now()
    print('Make stop words and Save succesfully! ',last-now,' time(s)')
def Load_Stop_Words():
    p=open('./data/stop_words','rb')
    data=pickle.load(p)
    stop_words=data['stop_words']
    print('Load stop words succesfully!')
    return stop_words
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

    stop_words=Load_Stop_Words()
    tfidf=TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=stop_words)
    x_train=tfidf.fit_transform(train_cut_words)
    x_test=tfidf.transform(test_cut_words)
    # dict_list=tfidf.get_feature_names()
    train_label_data=pd.read_csv('./data/train_label_remove_less_than.txt',names=['c1'])
    y_train=np.array(train_label_data['c1'])
    test_data = pd.read_csv('./data/test_text.csv', names=['data', 'label'])
    y_test = np.array(test_data['label'])

    #-----------------save----------------

    p={'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test}
    temp=open('./data/train_and_test_data_70000','wb')
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
def train_by_partial_SGD(filename):

    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf(filename)
    model= SGDClassifier(n_jobs=4,loss='hinge',alpha=0.09,tol=0.001)
    now = datetime.datetime.now()
    print("Training begin by SGB:", now)
    batch_size=50000
    for i in range(1000):
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
    filename='train_and_test_data_10000'
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf(filename)
    model = RandomForestClassifier(n_jobs=8,n_estimators=30)
    now = datetime.datetime.now()
    print("Training begin by RandForest:", now)

    # params = {'max_depth': list(range(2, 7)), 'n_estimators': list(range(10, 300, 10))}
    # gs = GridSearchCV(model, params, n_jobs=-1, cv=3, verbose=1)
    # gs.fit(x_train,y_train)
    # y_pre = gs.predict(x_test)
    # print(gs.score(x_test, y_test))

    model.fit(x_train,y_train)
    y_pre = model.predict(x_test)
    print(model.score(x_test, y_test))

    print(accuracy_score(y_test, y_pre))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)
def train_by_GBC():
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf()
    model = GradientBoostingClassifier()
    now = datetime.datetime.now()
    print("Training begin by Gradient Boost :", now)
    model.fit(x_train,y_train)
    y_pre = model.predict(x_test)
    print(model.score(x_test, y_test))

    print(accuracy_score(y_test, y_pre))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)
def train_by_xgboost_gpu():
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf()
    num_round = 10
    param = {'objective':'binary:logistic',  # Specify multiclass classification
             'max_depth':2, 'eta':1, 'silent':1,  # Number of possible output classes
             'tree_method': 'gpu_hist'  # Use GPU accelerated algorithm
             }
    # Convert input data from numpy to XGBoost format
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    now = datetime.datetime.now()
    print('Training begin:',now)
    bst=xgb.train(param, dtrain, num_round)
    y_pre=bst.predict(dtest)
    print(accuracy_score(y_test, y_pre))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)
def train_by_xgboost():
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf()
    model = XGBClassifier(n_jobs=10)
    now = datetime.datetime.now()
    print("Training begin by XGboost:", now)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(model.score(x_test, y_test))
    print(accuracy_score(y_test, y_pre))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)
def train_by_Kmean(filename):
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf(filename)
    now = datetime.datetime.now()
    print("Training begin by Kmeans:", now)
    model=MiniBatchKMeans(n_clusters=3150,batch_size=50000,verbose=1)
    model.partial_fit(x_train)
    print(model.score(x_test))
    training_time = datetime.datetime.now() - now
    print("Training time(s):", training_time)
# Load_Original_Traindata_Testdata_Cut_and_Save()
# Make_Stop_Words_and_Save(dimensions=70000)
# Make_Train_and_Test_Tf_Idf_and_Save()

filename='train_and_test_data_10000'

# train_by_RandForest()
# train_by_GBC()
# train_by_partial_SGD(filename)
# train_by_xgboost_gpu()
# train_by_xgboost()
train_by_Kmean(filename)