# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:39:23 2018

@author: Figo
"""

#使用数据增强解决数据少产生的过拟合的问题
#过拟合的原因是学习样本太少，导致无法训练出能够泛化到新数据的模型。
#数据增强是从现有的训练样本中生成更多的训练数据，
#其方法是利用多种能够生成可信图像的随机变换来增加（augment）样本。
#其目标是，模型在训练时不会两次查看完全相同的图像。
#这让模型能够观察到数据的更多内容，从而具有更好的泛化能力。

import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers

model_dir = 'D:/3_other_code/data/kaggle/model/'

plt_dir = 'D:/3_other_code/data/kaggle/result_plt/'

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


#为了进一步降低过拟合，你还需要向模型中添加一个 Dropout 层，添加到密集连接分类器之前

#定义一个包含 dropout 的新卷积神经网络

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten层用来将输入“压平”，即把多维的输入一维化，
# 常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])



#利用 ImageDataGenerator 来设置数据增强
#rotation_range 是角度值（在 0~180 范围内），表示图像随机旋转的角度范围。
#width_shift 和 height_shift 是图像在水平或垂直方向上平移的范围（相对于总宽度或总高度的比例）。
#shear_range 是随机错切变换的角度。
#zoom_range 是图像随机缩放的范围。
#horizontal_flip 是随机将一半图像水平翻转。如果没有水平不对称的假设（比如真实世界的图像），这种做法是有意义的
#fill_mode是用于填充新创建像素的方法，这些新像素可能来自于旋转或宽度/高度平移。

#利用数据增强生成器训练卷积神经网络
# 将训练集数据增强
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)
# 注意，不能增强验证数据
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')
# 训练及保存模型
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)
model.save(model_dir + 'cats_and_dogs_small_dropout.h5')

#绘制 使用了数据增强和 dropout 之后的模型的结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_dropout_acc')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_dropout_loss')
plt.show()