# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:23:50 2018

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

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

#对于用于特征提取的冻结的模型基，微调是指将其顶部的几层“解冻”，
#并将这解冻的几层和新增加的部分联合训练

#卷积基中更靠底部的层编码的是更加通用的可复用特征，而更靠顶部的层编码的是更专
#业化的特征。微调这些更专业化的特征更加有用，因为它们需要在你的新问题上改变用途。
#因此，在这种情况下，一个好策略是仅微调卷积基最后的两三层。

#冻结直到某一层的所有层
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    # set_trainable = True 之后，就能设置 trainable
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# 将部分解冻的卷积基和自定义的全连接网络分类器配置在一起
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 利用数据增强生成训练和验证数据
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


#微调模型
#我们将使用学习率非常小的 RMSProp 优化器来实现。之所以让学习率很小，
#是因为对于微调的三层表示，我们希望其变化范围不要太大。太大的权重更新可能会破坏这些表示。
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)
model.save(model_dir + 'cats_and_dogs_small_pretrained_fine_tuning.h5')


# 绘制结果
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_pretrained_fine_tuning_1_acc')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_pretrained_fine_tuning_1_loss')
plt.show()

# 上面这些曲线看起来包含噪声。为了让图像更具可读性，
#你可以将每个损失和精度都替换为指数移动平均值，从而让曲线变得平滑。
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs,smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_pretrained_fine_tuning_2_acc')
plt.figure()
plt.plot(epochs,smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_pretrained_fine_tuning_2_loss')
plt.show()


# 测试数据集上评估这个模型
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
                                                  class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)