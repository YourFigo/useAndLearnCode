# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:53:45 2018

@author: Figo
"""

import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'D:/3_other_code/data/kaggle/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

#利用 ImageDataGenerator 来设置数据增强
#rotation_range 是角度值（在 0~180 范围内），表示图像随机旋转的角度范围。
#width_shift 和 height_shift 是图像在水平或垂直方向上平移的范围（相对于总宽度或总高度的比例）。
#shear_range 是随机错切变换的角度。
#zoom_range 是图像随机缩放的范围。
#horizontal_flip 是随机将一半图像水平翻转。如果没有水平不对称的假设（比如真实世界的图像），这种做法是有意义的
#fill_mode是用于填充新创建像素的方法，这些新像素可能来自于旋转或宽度/高度平移。
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

from keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
#选择一张图像进行增强
img_path = fnames[3]
#读取图像并调整大小
img = image.load_img(img_path, target_size=(150, 150))
#将其转换为形状 (150, 150, 3) 的 Numpy 数组
x = image.img_to_array(img)
#将其形状改变为 (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)
#生成随机变换后的图像批量。循环是无限的，因此你需要在某个时刻终止循环
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()