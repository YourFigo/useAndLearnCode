# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:33:29 2018

@author: Figo
"""
import os

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


#### 将猫狗分类的小型卷积神经网络实例化
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 配置模型用于训练
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

#### 数据预处理
#(1) 读取图像文件。
#(2) 将 JPEG 文件解码为 RGB 像素网格。
#(3) 将这些像素网格转换为浮点数张量。
#(4) 将像素值（0~255 范围内）缩放到 [0, 1] 区间（正如你所知，神经网络喜欢处理较小的输入值）

# 使用 ImageDataGenerator 从目录中读取图像
from keras.preprocessing.image import ImageDataGenerator
# 将所有图像乘以 1/255 缩放
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# 将所有图像的大小调整为 150×150
# 因为使用了 binary_crossentropy损失，所以需要用二进制标签
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

#利用批量生成器拟合模型
# steps_per_epoch:每一轮需要从生成器中抽取多少个样本,steps_per_epoch*batch_size=2000
history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)
model.save(model_dir + 'cats_and_dogs_small_base.h5')

#绘制训练过程中的损失曲线和精度曲线
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
plt.savefig(plt_dir + 'cat_dog_convnet_base_acc')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'cat_dog_convnet_base_loss')
plt.show()

#由图可以看出，验证损失仅在 5 轮后就达到最小值，然后保持不变，
#而训练损失则一直线性下降，直到接近于 0。
# 因此下面开始处理过拟合

#使用数据增强解决数据少产生的过拟合的问题
#过拟合的原因是学习样本太少，导致无法训练出能够泛化到新数据的模型。
#数据增强是从现有的训练样本中生成更多的训练数据，
#其方法是利用多种能够生成可信图像的随机变换来增加（augment）样本。
#其目标是，模型在训练时不会两次查看完全相同的图像。
#这让模型能够观察到数据的更多内容，从而具有更好的泛化能力。


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
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()