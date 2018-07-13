import pandas as pd
import numpy as np
import re
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def Load_Original_Traindata_Testdata_Cut():
    word_string=''
    train_cut_words=[]
    for line in open('./data/train_text.txt',encoding='utf-8'):
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
    for (k,v) in c.most_common(10):
        most_common_words.append(k)
    stop_words = [item for item in all_words if item not in most_common_words]

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

def Load_Traindata_Testdata_with_Tfidf():
    p = open('./data/train_and_test_data', 'rb')
    data = pickle.load(p)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    return x_train,x_test,y_train,y_test


# train_cut_words,test_cut_words,stop_words=Load_Original_Traindata_Testdata_Cut()
# Tf_Idf_Save(train_cut_words,test_cut_words,stop_words)
x_train,x_test,y_train,y_test=Load_Traindata_Testdata_with_Tfidf()
