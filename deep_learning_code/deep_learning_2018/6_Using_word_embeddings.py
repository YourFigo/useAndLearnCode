# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:28:04 2018

@author: Figo
"""

################# 将一个 Embedding 层实例化  ########################
from keras.layers import Embedding
# Embedding 层至少需要两个参数：
# 标记的个数（这里是 1000，即最大单词索引 +1）和嵌入的维度（这里是 64）
embedding_layer = Embedding(1000, 64)

# 最好将 Embedding 层理解为一个字典，它接收整数作为输入，
# 并在内部字典中查找这些整数，然后返回相关联的向量。
# Embedding层实际上是一种字典查找： 单词索引 ---> Enbedding层 ---> 对应的词向量

##############加载 IMDB 数据，准备用于 Embedding 层 #################
from keras.datasets import imdb
from keras import preprocessing

# 作为特征的单词个数
max_features = 10000
# 在这么多单词后截断文本（这些单词都属于前 max_features 个最常见的单词）
maxlen = 20

# 将数据加载为整数列表 shape为(25000,)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#print(x_train.shape)
#print(x_train)


# 将整数列表转换成形状为(samples,maxlen) 的二维整数张量 shape为：(25000,20)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

################### 在 IMDB 数据上使用 Embedding 层和分类器 ##############
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# 指定 Embedding 层的最大输入长度，以便后面将嵌入输入展平。 
# Embedding 层激活的形状为 (samples, maxlen, 8)
# 该层每次输入10000条评论，每条评论的长度由input_length指定，转换为词向量的维度为8
model.add(Embedding(10000, 8, input_length=maxlen))

# 将三维的嵌入张量展平成形状为 (samples, maxlen * 8) 的二维张量
model.add(Flatten())

# 在上面添加分类器
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()


history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

#得到的验证精度约为 76%，考虑到仅查看每条评论的前 20 个单词，这个结果还是相当不错
#的。但请注意，仅仅将嵌入序列展开并在上面训练一个 Dense 层，会导致模型对输入序列中的
#每个单词单独处理，而没有考虑单词之间的关系和句子结构（举个例子，这个模型可能会将 this
#movie is a bomb 和 this movie is the bomb 两条都归为负面评论 a）。更好的做法是在嵌入序列上添
#加循环层或一维卷积层，将每个序列作为整体来学习特征。

