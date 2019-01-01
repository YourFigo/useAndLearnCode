# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:58:56 2018

@author: Figo
"""

from keras.models import load_model
# 导入保存的模型
model = load_model('D:/3_other_code/data/kaggle/model/cats_and_dogs_small_dropout.h5')
#print(model.summary())

plt_dir = 'D:/3_other_code/data/kaggle/visualization/dropout2/'

# 获取一张未参与训练的图片并进行预处理
img_path = 'D:/3_other_code/data/kaggle/cats_and_dogs_small/test/cats/cat.1700.jpg'
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
# expand_dims()进行维度扩展，axis=0，就在弟零个维度上进行扩展想·
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
#print(img_tensor.shape)

# 显示测试图像
#import matplotlib.pyplot as plt
#plt.imshow(img_tensor[0])
#plt.show()

# 用输入张量和输出张量来实例化一个model
from keras import models
# 提取前 8 层的输出
layer_outputs = [layer.output for layer in model.layers[:8]]
# 创建一个模型，给定模型输入，可以返回这些输出
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 以预测模式运行模型
# 返回 8个 Numpy数组组成的列表，每个层激活对应一个 Numpy 数组
activations = activation_model.predict(img_tensor)

# 取出第一层激活
#first_layer_activation = activations[0]
#print(first_layer_activation.shape)

#可视化第一层的第四个通道
import matplotlib.pyplot as plt
# 可视化第一层的第4和第7通道
#plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
#plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')

#将每个中间激活的所有通道可视化
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# zip()函数返回一个以元组为元素的列表
for layer_name, layer_activation in zip(layer_names, activations):
    # 每层的通道数
    n_features = layer_activation.shape[-1]
    # 特征图的形状为 (1, size, size, n_features)
    size = layer_activation.shape[1]
    # 每层需要显示图片的行数
    n_cols = n_features // images_per_row
    # 定义需要显示区域的总大小(高，宽)
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            # 获取某一个通道的输出
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            # 对某个通道的输出进行处理，使其美观
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # 显示每一个通道的网格
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    # 设置每个小图（通道图）的大小
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig(plt_dir + layer_name + '.png')