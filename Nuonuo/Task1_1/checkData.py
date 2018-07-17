import pickle
import numpy as np
def Load_Traindata_Testdata_with_Tfidf():
    p = open('./data/train_and_test_data', 'rb')
    data = pickle.load(p)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    print('Load training data succesfully!')
    return x_train,x_test,y_train,y_test

x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf()

print(x_train.shape)
# print(np.unique(y_train))