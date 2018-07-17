import pandas as pd
import numpy as np
# train_label_data = pd.read_csv('./data/train_label.txt', names=['c1'])
train_label_data = pd.read_csv('./data/train_label_remove_less_than.txt', names=['c1'])
y_train = np.array(train_label_data['c1'])
y_count = np.bincount(y_train)
# less_than_label=np.where(y_count<50)[0]
print('original labes:',y_train,'len:',len(y_train))
print('class labels:',np.unique(y_train))
print('samples in each class:',y_count)
# print('less than label:',less_than_label)