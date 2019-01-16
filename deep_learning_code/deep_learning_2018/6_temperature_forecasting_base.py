# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:32:42 2019

@author: Figo
"""

# 观察耶拿天气数据集的数据
import os

data_dir = 'D:/3_other_code/data/jena_climate/'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

# 解析数据,将 420 551 行数据转换成一个 Numpy 数组。
import numpy as np

# 忽略掉时间列了
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    # 每一行从第一列开始，忽略掉时间列
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
    
# 绘制温度时间序列,比如，温度随时间的变化
from matplotlib import pyplot as plt

temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)
plt.show()

# 绘制前 10 天的温度时间序列
plt.plot(range(1440), temp[:1440])
plt.show()