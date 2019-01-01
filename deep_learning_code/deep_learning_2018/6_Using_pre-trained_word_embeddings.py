# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 14:44:33 2018

@author: Figo
"""

# 输出路径
model_dir = 'D:/3_other_code/data/IMDB/model/'
plt_dir = 'D:/3_other_code/data/IMDB/result_plt/'

# 使用预训练词嵌入

####################处理 IMDB 原始数据的标签########################
import os

imdb_dir = 'D:/3_other_code/data/IMDB/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    # 将数据写入列表
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            # 原来没有 'r', encoding='UTF-8'  会报错
            # UnicodeDecodeError: 'gbk' codec can't decode byte 0x93
            # in position 130: illegal multibyte sequence
            f = open(os.path.join(dir_name, fname),'r', encoding='UTF-8')
            texts.append(f.read())
            f.close()
            # 将标签信息写入列表
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

#################对 IMDB 原始数据的文本进行分词#######################
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100  # 在 100 个单词后截断评论
training_samples = 200  # 在 200 个样本上训练
validation_samples = 10000  # 在 10 000 个样本上验证
max_words = 10000  # 只考虑数据集中前 10 000 个最常见的单词

tokenizer = Tokenizer(num_words=max_words)
# 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。
# texts 就是所有评价信息组成的列表，每个元素为一个评价信息。
tokenizer.fit_on_texts(texts)
# 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)]
sequences = tokenizer.texts_to_sequences(texts)

# word_index 一个dict，保存数据中所有word对应的编号id，从1开始
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# 将序列填充到 maxlen 长度
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将数据划分为训练集和验证集，但首先要打乱数据，
# 因为一开始数据中的样本是排好序的（所有负面评论都在前面，然后是所有正面评论）
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

#################解析 GloVe 词嵌入文件########################

glove_dir = 'D:/3_other_code/data/IMDB/glove.6B/'
# 我们对解压后的一个 .txt文件 进行解析，构建一个将单词（字符串）映射为其向量表示（数值向量）的索引。
# 每个单词对应一个数字向量    
# glove.6B.100d.txt 是 100 维的
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),'r',encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

################### 准备 GloVe 词嵌入矩阵 ##########################
# 这一步的目的是： 将数据word_index 的前 max_words 个词对应于 embeddings_index
# 的非空 嵌入向量 写入到 embedding_matrix 中，组成嵌入向量（用来加载进 Embedding 层中）。

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    # 对我们数据中的每个词，找出该词 在 GloVe 中词嵌入向量
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        # 只将前 max_words 个词的非空 嵌入向量 写入到 embedding_matrix 中
        if embedding_vector is not None:
            # 嵌入索引（embeddings_index）中找不到的词，其嵌入向量全为 0
            embedding_matrix[i] = embedding_vector

########################## 模型定义 ################################

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()

############## 将预训练的词嵌入加载到 Embedding 层中 ###################
# Embedding 层只有一个权重矩阵，是一个二维的浮点数矩阵，其中每个元素 i 是与索引 i相关联的词向量。

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.summary()

######################### 训练与评估 ############################
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights(model_dir + 'pre_trained_glove_model.h5')

######################## 绘制结果 #################################

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
plt.savefig(plt_dir + 'pre_trained_glove_acc')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'pre_trained_glove_loss')

plt.show()

# 从上面的输出情况看，很快出现了过拟合，考虑尝试不使用预训练词嵌入
############# 在不使用预训练词嵌入的情况下，训练相同的模型 #################
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save_weights(model_dir + 'no_using_pre_trained_glove_model.h5')


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
plt.savefig(plt_dir + 'no_using_pre_trained_glove_acc')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(plt_dir + 'no_using_pre_trained_glove_loss')

plt.show()

############## 对测试集数据进行分词、评估模型 ###################
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),'r', encoding='UTF-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

model.load_weights(model_dir + 'pre_trained_glove_model.h5')
test_loss, test_acc = test_result = model.evaluate(x_test, y_test)
print(test_loss,test_acc)