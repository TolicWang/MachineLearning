import pandas as pd
import numpy as np

label = pd.read_csv('./label.txt', names=['c1'])
y_train = np.array(label['c1'])
y_count = np.bincount(y_train)
less_than_label=np.where(y_count<5)[0]
print('original labes:',y_train)
print('class labels:',np.unique(y_train))
print('samples in each class:',y_count)
print('less than label:',less_than_label)

f = open('./x_train.txt', 'w', encoding='utf-8')
p = open('./y_train.txt', 'w', encoding='utf-8')
line_number=0
for line in open('./data.txt',encoding='utf-8'):
    if y_train[line_number] in less_than_label:
        line_number+=1
        continue
    line=line.strip('\n')

    f.write(line+ '\n')
    p.write(str(y_train[line_number]) + '\n')
    line_number+=1


f.close()
p.close()
