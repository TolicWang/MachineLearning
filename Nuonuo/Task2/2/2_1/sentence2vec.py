import pickle
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
def sentence2vec():
    TRAIN_NUM = 738793
    TEST_NUM = 13421
    DATA_SIZE = 128
    print('Load words vector!  ',datetime.datetime.now())
    p = open('../data/word_vec', 'rb')
    data = pickle.load(p)
    words=data['word']# dictionary
    vector = data['vec']
    word_list = []
    for key in words:
        word_list.append(words[key])
     #--------------------------train vector---------------------------------
    print('Make train sentence vector!  ', datetime.datetime.now())
    train_vec = np.zeros((TRAIN_NUM,DATA_SIZE),dtype=np.float32)
    for i,line in enumerate(open('../data/train_cut_words.txt',encoding='utf-8')):
        line = line.strip('\n').split()
        index =[]
        for word in line:
            if word in word_list:
                index.append(word_list.index(word))
        if len(index) == 0:
            temp = vector[0,:]
        else:
            temp = np.mean(vector[index,:],axis=0)
        train_vec[i,:] = temp
    p = {'train_vec': train_vec}
    temp = open('../data/train_vec', 'wb')
    pickle.dump(p, temp)
    print('Finished train_vec!  ',datetime.datetime.now())
    # --------------------------test vector---------------------------------
    print('Make test sentence vector!  ', datetime.datetime.now())
    test_vec = np.zeros((TEST_NUM, DATA_SIZE), dtype=np.float32)
    for i, line in enumerate(open('../data/test_cut_words.txt', encoding='utf-8')):
        line = line.strip('\n').split()
        index = []
        for word in line:
            if word in word_list:
                index.append(word_list.index(word))
        if len(index) == 0:
            temp = vector[0, :]
        else:
            temp = np.mean(vector[index, :], axis=0)
        test_vec[i, :] = temp
    p = {'test_vec': test_vec}
    temp = open('../data/test_vec', 'wb')
    pickle.dump(p, temp)
    print('Finished test_vec!  ',datetime.datetime.now())
def Load_train_and_test_data():
    p = open('../data/train_vec', 'rb')
    data = pickle.load(p)
    x_train = data['train_vec']
    train_label_data=pd.read_csv('../data/train_label_remove_less_than.txt',names=['c1'])
    y_train=np.array(train_label_data['c1'])

    p = open('../data/test_vec', 'rb')
    data = pickle.load(p)
    x_test = data['test_vec']
    test_data = pd.read_csv('../data/test_text.csv', names=['data', 'label'])
    y_test = np.array(test_data['label'])
    return x_train,y_train,x_test,y_test

# sentence2vec()
# x_train,y_train,x_test,y_test = Load_train_and_test_data()
