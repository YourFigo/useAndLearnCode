# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# 准备数据
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#print(train_images.shape)
#print(train_labels)
#print(test_images.shape)
#print(test_labels)


# 构建网络
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# 处理图像数据
print(train_images)
print('---------------------')
train_images = train_images.reshape((60000, 28 * 28))
print(train_images)
print('---------------------')
train_images = train_images.astype('float32') / 255
print(train_images)

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 准备标签
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 测试
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:{}  test_loss:{}'.format(test_acc,test_loss))