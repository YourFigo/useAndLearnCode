# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:06:14 2018

@author: Figo
"""

####对于在 ImageNet上预训练的 VGG16网络，其block3_conv1层第 0个过滤器激活的损失如下所示。######

#为过滤器的可视化定义损失张量
from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet',include_top=False)

'''
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# 为了实现梯度下降，我们需要得到损失相对于模型输入的梯度。
# 调用 gradients 返回的是一个张量列表（本例中列表长度为 1）。
# 因此，只保留第一个元素，它是一个张量
grads = K.gradients(loss, model.input)[0]

# 为了让梯度下降过程顺利进行，一个非显而易见的技巧是将梯度张量除以其 L2 范数（张量中
# 所有值的平方的平均值的平方根）来标准化。这就确保了输入图像的更新大小始终位于相同的范围。
# 做除法前加上 1e–5，以防不小心除以 0
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# 现在你需要一种方法：给定输入图像，它能够计算损失张量和梯度张量的值。
# 可以定义一个 Keras 后端函数来实现此方法： iterate 是一个函数，它将一个 Numpy 张量（表示为长度
# 为 1 的张量列表）转换为两个 Numpy 张量组成的列表，这两个张量分别是损失值和梯度值。
iterate = K.function([model.input], [loss, grads])
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

#通过随机梯度下降让损失最大化 
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128
step = 1.
for i in range(40):
    # 计算损失值和梯度值
    loss_value, grads_value = iterate([input_img_data])
    # 沿着让损失最大化的方向调节输入图像
    input_img_data += grads_value * step

'''

# 得到的图像张量是形状为 (1, 150, 150, 3) 的浮点数张量，其取值可能不是 [0, 255] 区间内的整数。
# 因此，你需要对这个张量进行后处理，将其转换为可显示的图像。
def deprocess_image(x):
    # 对张量做标准化，使其均值为 0，标准差为 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # 将 x 裁切（clip）到 [0, 1] 区间
    x += 0.5
    x = np.clip(x, 0, 1)
    # 将 x 转换为 RGB 数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 接下来，我们将上述代码片段放到一个 Python函数中，输入一个层的名称和一个过滤器索引，
# 它将返回一个有效的图像张量，表示能够将特定过滤器的激活最大化的模式。
# 生成过滤器可视化的函数
def generate_pattern(layer_name, filter_index, size=150):
    # 构建一个损失函数，将该层第 n 个过滤器的激活最大化
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # 计算这个损失相对于输入图像的梯度
    grads = K.gradients(loss, model.input)[0]
    # 标准化技巧：将梯度标准化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # 返回给定输入图像的损失和梯度
    iterate = K.function([model.input], [loss, grads])
    # 从带有噪声的灰度图像开始
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    # 运行 40 次梯度上升
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

# plt.imshow(generate_pattern('block3_conv1', 0))

# 我们只查看 VGG 每一层的前 64 个过滤器，并只查看每个卷积块的第一层（即 block1_conv1、
# block2_conv1、block3_conv1、 block4_ conv1、 block5_conv1）
# 我们将输出放在一个 8×8 的网格中，每个网格是一个 64 像素×64 像素的过滤器模式，
# 两个过滤器模式之间留有一些黑边

layer_name = ['block{}_conv1'.format(x) for x in range(1,6)]
layer_name = 'block4_conv1'
size = 64
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
        vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(results)