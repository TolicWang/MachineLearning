import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jieba
import re
from collections import Counter
def samples():
    # train_label_data = pd.read_csv('./data/train_label.txt', names=['c1'])
    test_data = pd.read_csv('./data/test_text.csv', names=['data', 'label'])
    y_train= np.array(test_data['label'])
    # y_train = np.array(train_label_data['c1'])
    print('样本总数：',len(y_train))
    x=np.unique(y_train)
    print('总类别：',x)
    print('总类别数',len(x))
    y=np.bincount(y_train)
    print('每个类的样本数：',y)
    index = np.where(y<20)[0]
    index0=np.where(y==0)[0]
    print('样本次数小于n的类别数：',len(index)-len(index0))
    s=y.sum()
    count=0
    for i in index:
        # s-=y[i]
        count+=y[i]
    print('样本次数小于n的个数%d,占比%f：'%(count,count/s))

    #
    # print(y_train)
    #
    # print(y)
    # print(index)
    #
    #

def words(dimensions):
    train_words=""
    for line in open('./data/train_cut_words.txt', encoding='utf-8'):
        line = line.strip('\n')+" "
        train_words+=line
    train_words=train_words.split()
    print(len(set(train_words)))
    c1 = Counter()
    for x in train_words:
        if len(x) > 1 and x != '\r\n':
            c1[x] += 1
    train_words_most = []
    for (k, v) in c1.most_common(dimensions):
        train_words_most.append(k)

    test_words=""
    for line in open('./data/test_cut_words.txt', encoding='utf-8'):
        line = line.strip('\n') + " "
        test_words += line
    test_words=list(set(test_words.split()))
    print(len((test_words)))

    # print(train_words_most)
    # print(test_words)

    # print('train_words_most_len:',len(train_words_most))
    # print('test_words_len:',len(test_words))
    unique_len = len(set(test_words + train_words_most))
    # print('unique_len:', unique_len)
    total_len = len(train_words_most) + len(test_words)
    # print('total_len:', total_len)
    repeat_len = total_len - unique_len
    # print('repeat_len:', repeat_len)
    print('repeat rate:%f, when common words is %d:'%(repeat_len / len(test_words),dimensions))

# for i in np.arange(200000,240001,20000,dtype=np.int32):
#     words(i)

# words(10)



A = ['A','B','C','AB','DF','C']
B = ['A','B','DF','H','OP','SD','H']

A = list(set(A))# len(A)=5
B = list(set(B))# len(B)=6

unique_len=len(set(A+B)) # 8
total_len=len(A)+len(B) # 11
repeat_len=total_len-unique_len # 3  A B DF
repeat_rate=repeat_len/len(A) # 3 / 5 = 0.6
print(repeat_rate)