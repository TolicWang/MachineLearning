import numpy as np
import pandas as pd
import scipy.io as load
import matplotlib.pyplot as plt
import  pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def costFandGradient(X,y_label,W1,b1,W2,b2,lambd):

    #============    forward propogation
    m, n = np.shape(X)  # m:samples, n: dimensions
    a1 = X.T  # 400 by 5000
    z2 = np.dot(W1, a1) + b1  # 25 by 400 * 400 by 5000 + 25 by 1= 25 by 5000
    a2 = sigmoid(z2)  # 25 by 5000
    z3 = np.dot(W2, a2) + b2  # 10 by 25 * 25 by 5000 + 10 by 1= 10 by 5000
    a3 = sigmoid(z3)  # 10 by 5000
    cost = (1/m)*np.sum(np.sum(-y_label*np.log(a3)-(1-y_label)*np.log(1-a3)))
    cost = cost+(lambd / (2*m)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    # print(cost)
    #===========   back propogation

    delta3=(a3-y_label)
    df_w2=np.dot(delta3,a2.T)# 10 by 5000 dot 5000 by 25 = 10 by 25
    df_w2=(1/m)*df_w2+(lambd/m)*W2

    delta2=np.dot(W2.T,delta3)*sigmoidGradient(z2)# 25 by 10 dot 10 by 5000 * 25 by 5000= 25 by 5000
    df_w1=np.dot(delta2,a1.T)# 25 by 5000 dot 5000 by 400 = 25 by 400
    df_w1=(1/m)*df_w1+(lambd/m)*W1

    df_b1=(1/m)*np.sum(delta2,axis=1).reshape(b1.shape)#
    df_b2=(1/m)*np.sum(delta3,axis=1).reshape(b2.shape)
    return cost,df_w1,df_w2,df_b1,df_b2

def gradientDescent(learn_rate,W1,b1,W2,b2,df_w1,df_w2,df_b1,df_b2):
    W1=W1-learn_rate*df_w1# 25 by 400
    W2=W2-learn_rate*df_w2# 10 by 25
    b1=b1-learn_rate*df_b1# 25 by 1
    b2=b2-learn_rate*df_b2# 10 by 1
    return W1,b1,W2,b2

def loadData():
    data=load.loadmat('ex4data1.mat')
    X=data['X']# 5000 by 400  samples by dimensions
    y=data['y'].reshape(5000)
    eye=np.eye(10)
    y_label=eye[:,y-1] # 10 by 5000
    ss=StandardScaler()
    X=ss.fit_transform(X)
    return X,y,y_label
def train():
    X, y, y_label=loadData()
    m,n=np.shape(X)# m:samples, n: dimensions
    input_layer_size=400
    hidden_layer_size=25
    output_layer_size=10
    epsilong_init=0.12
    W1=np.random.rand(hidden_layer_size,input_layer_size)*2*epsilong_init-epsilong_init
    W2=np.random.rand(output_layer_size,hidden_layer_size)*2*epsilong_init-epsilong_init
    b1=np.random.rand(hidden_layer_size,1)*2*epsilong_init-epsilong_init
    b2=np.random.rand(output_layer_size,1)*2*epsilong_init-epsilong_init

    lambd = 3.0
    iteration=200
    cost=[]
    learn_rate=1.7
    for i in range(iteration):
        arr = np.arange(5000)
        np.random.shuffle(arr)
        index = arr[:500]
        batch_X=X[index,:]
        batch_y=y_label[:,index]
        c,df_w1, df_w2, df_b1, df_b2=costFandGradient(batch_X, batch_y, W1, b1, W2, b2, lambd)
        cost.append(round(c,4))
        W1, b1, W2, b2=gradientDescent(learn_rate,W1,b1,W2,b2,df_w1,df_w2,df_b1,df_b2)
    p={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    temp=open('data','wb')
    pickle.dump(p,temp)

    x=np.arange(1,iteration+1)
    plt.plot(x,cost)
    plt.show()

def prediction():
    X, y, y_label = loadData()

    p = open('data', 'rb')
    data = pickle.load(p)
    W1 = data['W1']
    W2 = data['W2']
    b1 = data['b1']
    b2 = data['b2']
    a1 = X.T  # 400 by 5000
    z2 = np.dot(W1, a1) + b1  # 25 by 400 * 400 by 5000 + 25 by 1= 25 by 5000
    a2 = sigmoid(z2)  # 25 by 5000

    z3 = np.dot(W2, a2) + b2  # 10 by 25 * 25 by 5000 + 10 by 1= 10 by 5000
    a3 = sigmoid(z3)  # 10 by 5000
    y_pre = np.zeros(a3.shape[1], dtype=int)
    for i in range(a3.shape[1]):
        col = a3[:, i]
        index = np.where(col == np.max(col))[0][0] + 1
        y_pre[i] = index
    print(accuracy_score(y,y_pre))


if __name__ == '__main__':
    train()
    prediction()