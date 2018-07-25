import datetime
import pandas as pd
import numpy as np
import jieba
import re
import collections
import tensorflow as tf
from six.moves import xrange
import random
import pickle
def Load_Original_Traindata_Testdata_Cut_and_Save():
    """
    该函数的作用是将原始训练集（train_text.txt），测试集（test_text.csv)进行分词处理
    并分别保存至train_cut_words.txt和test_cut_words.txt
    其中：
    若去掉样本出现次数小于SAMPLE_FREQUENT的样本，则此时训练集对应的标签保存在
    train_label_remove_less_than.txt中；
    若SAMPLE_FREQUENT=1，则train_label_remove_less_than.txt等同于train_label.txt
    :return:
    """
    now=datetime.datetime.now()
    print("Train and Test words Cut begin: ",now)
    train_label_data = pd.read_csv('../data/train_label.txt', names=['c1'])
    y_train = np.array(train_label_data['c1'])
    y_count = np.bincount(y_train)
    SAMPLS_FREQUENT = 1
    less_than_label = np.where(y_count <SAMPLS_FREQUENT)[0]
    line_number = 0
    count=0
    f=open('../data/train_cut_words.txt','w',encoding='utf-8')
    p=open('../data/train_label_remove_less_than.txt','w',encoding='utf-8')
    for line in open('../data/train_text.txt',encoding='utf-8'):
        if y_train[line_number] in less_than_label:
            line_number += 1
            continue
        p.write(str(y_train[line_number])+'\n')
        line_number += 1
        count+=1
        line = line.strip('\n')
        # line = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", line)
        line = re.sub("[0-9\_\-\\\/\#\(\) \（ \）\!\%\[\]\、\,\。\*\+\【\】\."
                      "\，\"\”\“\Ф\×\℃\》]", "", line)
        line = line.encode('utf-8').decode('utf-8-sig')
        seg_list=jieba.cut(line,cut_all=False)
        temp=" ".join(seg_list)
        f.write(temp+'\n')

    f.close()
    p.close()
    test_data = pd.read_csv('../data/test_text.csv', names=['data', 'label'])
    test_datas = np.array(test_data['data'])
    f=open('../data/test_cut_words.txt','w',encoding='utf-8')
    for line in test_datas:
        line = re.sub("[0-9\_\-\\\/\#\(\) \（ \）\!\%\[\]\、\,\。\*\+\【\】\."
                      "\，\"\”\“\Ф\×\℃\》]", "", line)
        seg_list = jieba.cut(line, cut_all=False)
        temp=" ".join(seg_list)
        f.write(temp+'\n')
    f.close()
    last=datetime.datetime.now()
    print('Train and Test words Cut and Save succesfully!',last-now," time(s)")

def Load_train_and_test_cut_words():
    now = datetime.datetime.now()
    print("Load train_cut_words begin: ", now)
    train_cut_words = []
    for line in open('../data/train_cut_words.txt', encoding='utf-8'):
        line = line.strip('\n')+" "
        line=line.split()
        line = [item for item in line if len(item) > 1]
        train_cut_words.extend(line)
    print("Load test_cut_words begin: ", now)
    test_cut_words = []
    for line in open('../data/test_cut_words.txt',encoding='utf-8'):
        line = line.strip('\n')+" "
        line=line.split()
        line = [item for item in line if len(item) > 1]
        test_cut_words.extend(line)

    all_words = train_cut_words + test_cut_words
    return  all_words

def build_dataset(words, n_words):
    """
    函数功能：将原始的单词表示变成index
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)# 给字典赋值,key:word  valuew:0-n_word-1
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # UNK的index为0
            unk_count += 1
        data.append(index)
        # data 用来记录words中每个单词在dictionary中的位置，若该单词未在dictionary中，则为0
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def generate_batch(data,batch_size, num_skips, skip_window):
    # 定义一个函数，用于生成skip-gram模型用的batch
    # data_index相当于一个指针，初始为0
    # 每次生成一个batch，data_index就会相应地往后推
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    # data_index是当前数据开始的位置
    # 产生batch后就往后推1位（产生batch）
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
     # 利用buffer生成batch
     # buffer是一个长度为 2 * skip_window + 1长度的word list
     # 一个buffer生成num_skips个数的样本
     #    print([reverse_dictionary[i] for i in buffer])
        target = skip_window  # target label at the center of the buffer
    #     targets_to_avoid保证样本不重复
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        # 每利用buffer生成num_skips个样本，data_index就向后推进一位
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def train(data):
    # 第四步: 建立模型.
    batch_size = 128
    embedding_size = 128  # 词嵌入空间是128维的。即word2vec中的vec是一个128维的向量
    skip_window = 1  # skip_window参数和之前保持一致
    num_skips = 2  # num_skips参数和之前保持一致

    # 在训练过程中，会对模型进行验证
    # 验证的方法就是找出和某个词最近的词。
    # 只对前valid_window的词进行验证，因为这些词最常出现
    valid_size = 16  # 每次验证16个词
    valid_window = 100  # 这16个词是在前100个最常见的词中选出来的
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    # 构造损失时选取的噪声词的数量
    num_sampled = 64

    graph = tf.Graph()

    with graph.as_default():
        # 输入的batch
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # 用于验证的词
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # 下面采用的某些函数还没有gpu实现，所以我们只在cpu上定义模型
        with tf.device('/cpu:0'):
            # 定义1个embeddings变量，相当于一行存储一个词的embedding
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            # 利用embedding_lookup可以轻松得到一个batch内的所有的词嵌入
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # 创建两个变量用于NCE Loss（即选取噪声词的二分类损失）
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / np.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # tf.nn.nce_loss会自动选取噪声词，并且形成损失。
        # 随机选取num_sampled个噪声词
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

    # 得到loss后，我们就可以构造优化器了
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # 计算词和词的相似度（用于验证）
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        # 找出和验证词的embedding并计算它们和所有单词的相似度
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # 变量初始化步骤
        init = tf.global_variables_initializer()

    # 第五步：开始训练
    num_steps = 100001

    with tf.Session(graph=graph) as session:
        # 初始化变量
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data,
                batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # 优化一步
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # 2000个batch的平均损失
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # 每1万步，我们进行一次验证
            if step % 10000 == 0:
                # sim是验证词与所有词之间的相似度
                sim = similarity.eval()
                # 一共有valid_size个验证词
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # 输出最相邻的8个词语
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        # final_embeddings是我们最后得到的embedding向量
        # 它的形状是[vocabulary_size, embedding_size]
        # 每一行就代表着对应index词的词嵌入表示
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings




data_index = 0
# Load_Original_Traindata_Testdata_Cut_and_Save()
vocabulary = Load_train_and_test_cut_words()
print('Data size', len(vocabulary)) # 总长度
vocabulary_size = 100000
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary
# 输出最常出现的5个单词
print('Most common words (+UNK)', count[:5])
# 输出转换后的数据库data，和原来的单词（前10个）
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# 我们下面就使用data来制作训练集
# 默认情况下skip_window=1, num_skips=2
# 此时就是从连续的3(3 = skip_window*2 + 1)个词中生成2(num_skips)个样本。
# 如连续的三个词['used', 'against', 'early']
# 生成两个样本：against -> used, against -> early
batch, labels = generate_batch(data=data,batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

final_embeddings=train(data)

p = {'word': reverse_dictionary,'vec':final_embeddings}
temp = open('../data/word_vec', 'wb')
pickle.dump(p, temp)




