# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:35:25 2019

@author: Figo
"""

'''
这个问题的确切表述如下：一个时间步是 10 分钟，每 steps 个时间步采样一次数据，给
定过去 lookback 个时间步之内的数据，能否预测 delay 个时间步之后的温度？用到的参数值如下。

lookback = 720：给定过去 5 天内的观测数据。
steps = 6：观测数据的采样频率是每小时一个数据点。
delay = 144：目标是未来 24 小时之后的数据。
'''

# 输出路径
model_dir = 'D:/3_other_code/data/jena_climate/model/'
plt_dir = 'D:/3_other_code/data/jena_climate/result_plt/'

############################# 准备数据 #################################

# 1 加载数据
'''
加载耶拿天气数据集，并处理为numpy格式数据
'''

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


# 2 数据标准化标准化，将每个时间序列减去其平均值，然后除以其标准差。
'''
数据已经是数值型的，所以不需要做向量化。
但数据中的每个时间序列位于不同的范围（比如温度通道位于 -20 到 +30之间，
但气压大约在 1000 毫巴上下）。你需要对每个时间序列分别做标准化，让它们在
相似的范围内都取较小的值。
'''
# 我们将使用前200 000 个时间步作为训练数据，所以只对这部分数据计算平均值和标准差。

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# 3 生成时间序列样本及其目标的生成器
'''
编写一个 Python 生成器，以当前的浮点数数组作为输入，并从最近的数据中生成数据批
量，同时生成未来的目标温度。因为数据集中的样本是高度冗余的（对于第 N 个样本和
第 N+1 个样本，大部分时间步都是相同的），所以显式地保存每个样本是一种浪费。相反，
我们将使用原始数据即时生成样本。
'''

# 它生成了一个元组 (samples, targets)，其中 samples 是输入数据的一个批量， targets 是对应的目标温度数组。
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    '''
    data：浮点数数据组成的原始数组
    lookback：输入数据应该包括过去多少个时间步
    delay：目标应该在未来多少个时间步之后
    min_index 和 max_index： data 数组中的索引，用于界定需要抽取哪些时间步。这有
    助于保存一部分数据用于验证、另一部分用于测试。
    shuffle：是打乱样本，还是按顺序抽取样本。
    batch_size：每个批量的样本数
    step：数据采样的周期（单位：时间步）。我们将其设为 6，为的是每小时抽取一个数据点。
    '''
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

# 4 准备训练生成器、验证生成器和测试生成器

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# 为了查看整个验证集，需要从 val_gen 中抽取多少次
val_steps = (300000 - 200001 - lookback) // batch_size

# 为了查看整个测试集，需要从test_gen 中抽取多少次
test_steps = (len(float_data) - 300001 - lookback) // batch_size


##################一种基于常识的、非机器学习的基准方法########################
'''
本例中，我们可以放心地假设，温度时间序列是连续的（明天的温度很可能接近今天的温
度），并且具有每天的周期性变化。因此，一种基于常识的方法就是始终预测 24 小时后的温度
等于现在的温度。我们使用平均绝对误差（MAE）指标来评估这种方法。
'''

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    
evaluate_naive_method()
# 得到的 MAE 为 0.29。因为温度数据被标准化成均值为 0、标准差为 1，所以无法直接对这
# 个值进行解释。它转化成温度的平均绝对误差为 0.29×temperature_std 摄氏度，即 2.57℃。
celsius_mae = 0.29 * std[1]

'''
###################### 一种基本的机器学习方法 ###########################

####### 尝试使用简单且计算代价低的机器学习模型--小型的密集连接网络

# 训练并评估一个密集连接模型
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
model.save_weights(model_dir + 'jena_climate_simply_fully-connected.h5')

# 绘制结果
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'jena_climate_simply_fully-connected_loss')

plt.show()
'''

######### 第一个循环网络基准

## 训练并评估一个基于 GRU 的模型
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
model.save_weights(model_dir + 'jena_climate_GRU_base.h5')

# 绘制结果
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'jena_climate_GRU_base_loss')

plt.show()

## 使用循环 dropout 来降低过拟合
