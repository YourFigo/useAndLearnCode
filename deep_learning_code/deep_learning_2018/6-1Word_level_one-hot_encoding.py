# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:18:28 2018

@author: Figo
"""

import numpy as np

# 初始化数据可以是一个句子或者一个文档
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 构建数据中所有标记的索引
token_index = {}
for sample in samples:
    # 利用 split 方法对样本进行分词。在实际应用中，
    # 还需要从样本中去掉标点和特殊字符
    for word in sample.split():
        # 为每个唯一单词指定一个唯一索引。
        if word not in token_index:
            # 注意，没有为索引编号 0 指定单词
            token_index[word] = len(token_index) + 1

# 对样本进行分词。只考虑每个样本前 max_length 个单词
max_length = 10
# 将结果保存在 results 中
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
