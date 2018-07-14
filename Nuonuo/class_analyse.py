import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_label_data = pd.read_csv('./data/train_label.txt', names=['c1'])
# train_label_data = pd.read_csv('./test', names=['c1'])
y_train = np.array(train_label_data['c1'])

num = 600
x=np.unique(y_train)
y=np.bincount(y_train)
index = np.where(y < 100)[0]

s=y.sum()
print(s)
for i in index:
    s-=y[i]
print(s)

print(y_train)
print(x)
print(y)
print(index)


