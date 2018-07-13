from collections import Counter
import numpy as np
a=['电话', '贴膜']
b=['客梯口', '电话', '贴膜', '鲜艳', '鲜艳', '鲜艳', '鲜艳客梯口', '电话', '贴膜客梯口', '电话', '贴膜', '鲜艳客梯口', '电话', '贴膜']


ret_list = [item for item in b if item not in a]
print(ret_list)
#
# for x in content:
#     if len(x) > 1 and x != '\r\n':
#         c[x] += 1
# print('常用词频度统计结果')
# listone = []
# for (k, v) in c.most_common(10):
#     listone.append(k)
#     print('%s:%d' % ( k,v))