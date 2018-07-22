# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
X_test = ['没有 你 的 地方 都是 他乡','没有 你 的 旅行 都是 流浪']

stopword = ['都是','都是'] #自定义一个停用词表，如果不指定停用词表，
# 则默认将所有单个汉字视为停用词；但可以设token_pattern=r"(?u)\b\w+\b"，即不考虑停用词

tfidf=TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",stop_words=stopword)
weight=tfidf.fit_transform(X_test).toarray()
word=tfidf.get_feature_names()

print('vocabulary list:\n')
for key,value in tfidf.vocabulary_.items():
    print (key,value)

print('IFIDF词频矩阵:\n')
print(weight)

# for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#     print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
#     for j in range(len(word)):
#         print (word[j], weight[i][j])#第i个文本中，第j个次的tfidf值
