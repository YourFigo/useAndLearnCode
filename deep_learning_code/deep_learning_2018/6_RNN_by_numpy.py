# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 21:22:14 2018

@author: Figo
"""

# 简单 RNN 的 Numpy 实现

import numpy as np

#输入序列的时间步数
timesteps = 100
#输入特征空间的维度
input_features = 32
#输出特征空间的维度
output_features = 64

# 输入数据：随机噪声，仅作为示例
inputs = np.random.random((timesteps, input_features))

# 初始状态：全零向量
state_t = np.zeros((output_features,))

# 创建随机的权重矩阵
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:     # input_t 是形状为 (input_features,) 的向量
    # 由输入和当前状态（前一个输出）计算得到当前输出
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    # 将这个输出保存到一个列表中
    successive_outputs.append(output_t)
    # 更新网络的状态，用于下一个时间步
    state_t = output_t

# 最终输出是一个形状为 (timesteps,output_features) 的二维张量
# np.stack() 增加一维，新维度的下标为 axis=0
final_output_sequence = np.stack(successive_outputs, axis=0)

print(len(successive_outputs))
print(len(successive_outputs[0]))
print(final_output_sequence.shape)