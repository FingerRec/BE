# from scipy import stats
# import os
#
# x = []
# y = []
# with open('pearson_calculate.txt', 'r') as f:
#     lines = f.readlines()
# for line in lines:
#     if len(line.strip().split(' '))==2:
#         x.append(float(line.strip().split(' ')[0]))
#         y.append(float(line.strip().split(' ')[1]))
#     else:
#         x.append(float(line.strip().split('\t')[0]))
#         y.append(float(line.strip().split('\t')[1]))
# print(len(x), len(y))
# print(stats.pearsonr(x, y))
#

from scipy import stats
import os

x = []
y = []
g = open('multi100.txt', 'w')
with open('pearson_calculate.txt', 'r') as f:
    lines = f.readlines()
for line in lines:
    if len(line.strip().split(' ')) is 2:
        g.write(str(float(line.strip().split(' ')[1])*100) + '\n')
    else:
        g.write(
            str(float(line.strip().split('\t')[1]) * 100) + '\n')