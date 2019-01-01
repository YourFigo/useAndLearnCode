# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:46:17 2018

@author: Figo
"""

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 创建一个分词器（tokenizer），设置为只考虑前 1000 个最常见的单词
tokenizer = Tokenizer(num_words=1000)
# 构建单词索引
tokenizer.fit_on_texts(samples)

# 将字符串转换为整数索引组成的列表
sequences = tokenizer.texts_to_sequences(samples)
# sequences 为 [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

# 也可以直接得到 one-hot 二进制表示。
# 这个分词器也支持除 one-hot 编码外的其他向量化模式
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# one_hot_results.shape 为 (2, 1000)

# 找回单词索引
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))