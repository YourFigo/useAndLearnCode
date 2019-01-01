# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:25:59 2018

@author: Figo
"""

import os
model_dir = 'D:/3_other_code/data/kaggle/model/'
plt_dir = 'D:/3_other_code/data/kaggle/result_plt/'

base_dir = 'D:/3_other_code/data/kaggle/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
base_dir = 'D:/3_other_code/data/kaggle/cats_and_dogs_small'

#使用在 ImageNet 上训练的 VGG16 网络的卷积基从猫狗图像中提取有趣的特征，
#然后在这些特征上训练一个猫狗分类器。

#将 VGG16 卷积基实例化
#weights 指定模型初始化的权重检查点。
#include_top 指定模型最后是否包含密集连接分类器。
#默认情况下，这个密集连接分类器对应于 ImageNet 的 1000 个类别。
#因为我们打算使用自己的密集连接分类器（只有两个类别： cat 和 dog），所以不需要包含它。
#input_shape 是输入到网络中的图像张量的形状。
#这个参数完全是可选的，如果不传入这个参数，那么网络能够处理任意形状的输入。
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# 我们这里使用 VGG16 ，最后的特征图形状为 (4, 4, 512)

########### 我们将在这个特征上添加一个密集连接分类器。有两种处理方法：
########### 1、在你的数据集上运行卷积基，将输出保存成硬盘中的 Numpy 数组，然后用这个数据作
########### 为输入，输入到独立的密集连接分类器中。
########### 2、 在顶部添加 Dense 层来扩展已有模型（即 conv_base），并在输入数据上端到端地运行
########### 整个模型。这样你可以使用数据增强，因为每个输入图像进入模型时都会经过卷积基。

############################### 方法1：###############################
#不使用数据增强的快速特征提取
#使用预训练的卷积基提取特征
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        # 注意，这些生成器在循环中不断生成数据，所以你必须在读取完所有图像后终止循环
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

#目前，提取的特征形状为 (samples, 4, 4, 512)。我们要将其输入到密集连接分类器中，
#所以首先必须将其形状展平为 (samples, 8192)。
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

#定义自己的密集连接分类器（注意要使用 dropout 正则化），并在刚刚保存的数据
#和标签上训练这个分类器。
from keras import models
from keras import layers
from keras import optimizers
model1 = models.Sequential()
model1.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(1, activation='sigmoid'))

model1.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history1 = model1.fit(train_features,
                    train_labels,epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

model1.save(model_dir + 'cats_and_dogs_small_pretrained_feature_extraction_1.h5')

# 绘制结果
import matplotlib.pyplot as plt
acc = history1.history['acc']
val_acc = history1.history['val_acc']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_pretrained_feature_extraction_1_acc')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_pretrained_feature_extraction_1_loss')
plt.show()
############################### 方法1：###############################


############################### 方法2：###############################
# 方法1 未使用数据增强，从acc曲线来看，从一开始就过拟合了，下面使用 数据增强
#使用数据增强的特征提取

#在卷积基上添加一个密集连接分类器
from keras import models
from keras import layers

model2 = models.Sequential()
# 在 Keras 中，冻结网络的方法是将其 trainable 属性设为 False。
# 冻结之前训练好的卷积基，防止训练时被修改,如此设置之后，只有添加的两个 Dense 层的权重才会被训练
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
#conv_base.trainable = False

model2.add(conv_base)
model2.add(layers.Flatten())
model2.add(layers.Dense(256, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))

#利用冻结的卷积基端到端地训练模型
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

model2.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history2 = model2.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)
model2.save(model_dir + 'cats_and_dogs_small_pretrained_feature_extraction_2.h5')

# 绘制结果
import matplotlib.pyplot as plt
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_pretrained_feature_extraction_2_acc')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_pretrained_feature_extraction_2_loss')
plt.show()

############################### 方法2：###############################