# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:33:28 2019

@author: Figo
"""

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

##########利用keras中的SimpleRNN层 测试全部时间步的输出和最后一个时间步的输出  ##############3

# 只返回最后一个时间步的输出。
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

# 返回每个时间步的输出
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

# 将多个循环层逐个堆叠 并 让所有中间层都返回完整的输出序列  最后一层仅返回最终输出
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
model.summary()
print('###################################################')

##############  将这个模型应用于 IMDB 电影评论分类问题 ###################

# 输出路径
model_dir = 'D:/3_other_code/data/IMDB/model/'
plt_dir = 'D:/3_other_code/data/IMDB/result_plt/'
      
# 准备 IMDB 数据
from keras.datasets import imdb
from keras.preprocessing import sequence

# 作为特征的单词个数
max_features = 10000
# 在这么多单词之后截断文本（这些单词都属于前 max_features 个最常见的单词）
maxlen = 500
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# 用 Embedding 层和 SimpleRNN 层来训练模型
from keras.layers import Dense

model = Sequential()
# 输入：(batch_size, maxlen)，输出：(batch_size, maxlen, output_dim)
# 32 用于指定Embedding层的输出维度
model.add(Embedding(max_features, 32))
# 输入：(batch_size, timesteps,input_features)，输出：(batch_size, output_features) 
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
model.save_weights(model_dir + 'Understanding_RNN_using_SimpleRNN.h5')

# 绘制结果
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(plt_dir + 'Understanding_RNN_using_SimpleRNN_acc')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'Understanding_RNN_using_SimpleRNN_loss')

plt.show()

##################使用 Keras 中的 LSTM 层######################
print('#####################################')

# 定义LSTM模型并训练
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
model.save_weights(model_dir + 'Understanding_RNN_using_LSTM.h5')

# 绘制 acc 和 loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(plt_dir + 'Understanding_RNN_using_LSTM_acc')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'Understanding_RNN_using_LSTM_acc')

plt.show()