import collections
import re
import jieba
words = ['液化气', '果味', '柚子', '绿茶', '住宿费', '果味', '柚子', '绿茶', '住宿费', '果味']
def build_dataset(words, n_words):
    """
    函数功能：将原始的单词表示变成index
    """
    print('words',words)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    print('count',count)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)# 给字典赋值,key:word  valuew:0-n_word-1; 给每个单词编号
    print('dictionary',dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # UNK的index为0
            unk_count += 1
        data.append(index)#words中每个单词在dictionary中的位置，若该单词未在dictionary中，则为0
    count[0][1] = unk_count
    print('data',data)
    print('count',count)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print('reversed_dictionary',reversed_dictionary)
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reversed_dictionary=build_dataset(words,3)

print('Sample data', data[:4], [reversed_dictionary[i] for i in data[:4]])