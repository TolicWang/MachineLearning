import pickle
import numpy as np
import datetime
from scipy import sparse as sp
import scipy as sp
from sklearn.metrics import accuracy_score

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
    n_centers = class_centers.shape[0]
    y_most = np.zeros(n_centers,dtype=np.float32)
    v2 = np.nan_to_num(v)
    for i in range(n_centers):
        v1 = class_centers[i]
        v1_dot_v2 = np.dot(v1,v2)
        v1_nor = np.sqrt(np.sum(v1*v1))
        v2_nor = np.sqrt(np.sum(v2*v2))
        y_most[i] = np.nan_to_num(v1_dot_v2/(v1_nor*v2_nor))
    # print(y_most)
    return np.argmax(y_most)
def Prediction(class_center,x_test,y_test):
    print(class_center.shape)
    now=datetime.datetime.now()
    print("Prediction begin: ",now)
    n_sampes = x_test.shape[0]
    y_pre = np.zeros(n_sampes,dtype=np.int32)
    for i in range(n_sampes):
        if (i+1) % 50 == 0:
            print("%d samples finished, %d left,acc %f   %s"
                  %(i,n_sampes-i,accuracy_score(y_pre[:i],y_test[:i]),datetime.datetime.now()))
        # s2 = np.power((class_center - x_test[i]),2)
        # s = np.sum(s2,axis=1)
        # y_pre[i] = np.argmin(s)
        y_pre[i] = cos_simi(class_center,x_test[i])
    last=datetime.datetime.now()
    print("Prediction finished!: ",last-now)
    return y_pre
def test_compute():
    x=np.linspace(1,50,50).reshape((10,5))
    y=np.array([0,0,1,1,2,2,3,3,2,3])
    print('training datas:',x)
    print('training labels:',y)
    class_center = Compute_Center_Each_Class(x,y,3)
    print('class centers:',class_center)
    x_test = np.array([[1,1,2,3,3,],
                       [10,11,12,9,16],
                       [10,11,11,9,17],
                       [10, 22, 31, 29, 37]])
    print('testing datas:', x_test)
    print('predicted labels:',Prediction(class_center,x_test))
def test_cos():
    c = np.array([[1,2,2,1],
                  [1,2,3,3],
                  [2, 1, 3, 3],])
    v = np.array([1,2,3,1])
    print(cos_simi(c,v))


if __name__ == '__main__':
    x_train, x_test, y_train, y_test=\
        Load_Traindata_Testdata_with_Tfidf('train_and_test_data_30000')
    class_center = Compute_Center_Each_Class(x_train,y_train,batch_size=30000)
    y_pre = Prediction(class_center,x_test.toarray(),y_test)
    print("accuracy:",accuracy_score(y_true=y_test,y_pred=y_pre))


    # print(cos_simi(v1,v2))
    # test_compute()



    # test_cos()


    # t=np.array([[0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0.4, 0.5, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0],
    #             [0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # s=sp.csr_matrix(t)

