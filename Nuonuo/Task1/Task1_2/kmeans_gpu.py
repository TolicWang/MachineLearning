import pickle
import datetime
import numpy as np
import tensorflow as tf

def Load_Traindata_Testdata_with_Tfidf(filename):
    p = open('../data/'+filename, 'rb')
    data = pickle.load(p)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    print('Load training data succesfully! shape:',x_train.shape)
    return x_train,x_test,y_train,y_test

def Compute_Center_Each_Class(x_train,y_train,batch_size):
    now=datetime.datetime.now()
    print("Compute center of each class begin: ",now)
    # x_train = x_train.toarray()
    classes = np.unique(y_train)# 0 1 2 ....3509
    class_center = np.zeros([len(classes),x_train.shape[1]])# 3510 by dimension
    times = int(x_train.shape[0]/batch_size) + 1
    for k in range(times):
        begin = k * batch_size
        end = begin + batch_size
        if end >= x_train.shape[0]:
            end = x_train.shape[0]
        batch_x = x_train[begin:end].toarray()
        # batch_x = x_train[begin:end]
        batch_y = y_train[begin:end]
        batch_classes = np.unique(batch_y)
        for i in batch_classes:
            index = np.where(batch_y == i)[0]
            samples = batch_x[index,:]
            class_center[i,:] += np.sum(samples,axis=0)
    c=np.bincount(y_train).reshape(class_center.shape[0],1)
    last=datetime.datetime.now()
    print("Compute center of each class finished!: ",last-now)
    return class_center/c
def cos_simi(class_centers,v):
    sess = tf.Session()
    class_centers = tf.convert_to_tensor_or_sparse_tensor(class_centers,dtype=tf.float32)
    v = tf.convert_to_tensor_or_sparse_tensor(v,dtype=tf.float32)
    n_centers = tf.shape(class_centers)[0]
    y_most =tf.Variable(tf.zeros(shape=tf.shape(n_centers),dtype=tf.float32,name='y_most'))
    v2 = tf.clip_by_value(v,1e-10,1.0)
    for i in range(n_centers.eval(session=sess)):
        v1 = class_centers[i]
        v1_dot_v2 = (v1*v2)
        v1_nor = tf.norm(v1)
        v2_nor = tf.norm(v2)
        y_most=y_most[i].assign( tf.clip_by_value((v1_dot_v2/(v1_nor*v2_nor)),1e-10,1.0))
    # print(y_most)
    return tf.argmax(y_most,1)


c = np.array([[1,2,2,1],
              [1,2,3,3],
              [2, 1, 3, 3]],dtype=np.float32)
v = np.array([1,2,3,1],np.float32)

m = cos_simi(c,v)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(m))