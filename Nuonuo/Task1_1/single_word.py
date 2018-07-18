import datetime
import pandas as pd
import numpy as np
import re
def Load_Original_Traindata_Testdata_Cut_and_Save():
    now=datetime.datetime.now()
    print("Train and Test words Cut begin: ",now)
    train_label_data = pd.read_csv('./data/train_label.txt', names=['c1'])
    y_train = np.array(train_label_data['c1'])
    y_count = np.bincount(y_train)
    less_than_label = np.where(y_count <1)[0]
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