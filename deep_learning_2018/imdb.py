# -*- coding: utf-8 -*-

from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

import numpy as np
# 将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
    return results

# 处理数据和标签
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(type(train_data))
print(train_data.shape)
print(train_data[0])
print(type(x_train))
print(x_train.shape)
print(x_train[0])

# 搭建网络
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
# 配置优化器
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

# 使用自定义的损失和指标
#from keras import losses
#from keras import metrics
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

#  留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 训练模型
#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history1 = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))
results1 = model.evaluate(x_test, y_test)
print(results1)

# 绘制训练损失和验证损失
import matplotlib.pyplot as plt
history_dict = history1.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制训练精度和验证精度
#plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 从图中可以看出，验证集的acc在第2epoch达到最大后，防止出现过拟合，所以减少训练次数
history2 = model.fit(partial_x_train,partial_y_train,epochs=4,batch_size=512,validation_data=(x_val, y_val))
results2 = model.evaluate(x_test, y_test)
print(results2)

# 训练好网络之后，你希望将其用于实践。你可以用 predict 方法来得到评论为正面的可能性大小。
print(model.predict(x_test))