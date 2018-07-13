import numpy as np
import pandas as pd
import re
import jieba
# f=open('./test_text.csv','w',encoding='utf8')
# for line in open("./test_text_label.txt",encoding='utf8'):
#     line=line.replace(',','')
#     line=line.replace('_label_',',')
#     f.write(line)

testData = pd.read_csv('./test_text.csv', names=['data', 'label'])
test_datas = np.array(testData['data'])
test_label = np.array(testData['label'])

test_cut_words=[]
for line in test_datas:
    line = re.sub("[\_\-\\\/\#\( \（ \）\!\%\[\]\、\,\。]", "", line)
    seg_list = jieba.cut(line, cut_all=False)
    test_cut_words.append(" ".join(seg_list))
