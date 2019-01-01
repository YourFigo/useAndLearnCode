# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:32:28 2018

@author: Figo
"""

import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
# 将可打印字符进行编号
token_index = dict(zip(characters, range(1, len(characters) + 1)))

# 分词只考虑前 50 个字符
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.