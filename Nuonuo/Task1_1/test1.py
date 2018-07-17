from collections import Counter
import numpy as np

# ret_list = [item for item in b if item not in a]
a=['地方','他乡','都是','他乡', '他乡','都是','都是']
print(a)
a=list(set(a))
print(a)
## for x in content:
#     if len(x) > 1 and x != '\r\n':
#         c[x] += 1
# print('常用词频度统计结果')
# listone = []
# for (k, v) in c.most_common(10):
#     listone.append(k)
#     print('%s:%d' % ( k,v))
