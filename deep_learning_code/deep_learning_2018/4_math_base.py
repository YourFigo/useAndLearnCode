# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:33:17 2018

@author: Figo
"""

import numpy as np
x = np.array(12)

x = np.array([12, 3, 6, 14, 7])

x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])

x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
            [[5, 78, 2, 34, 0],
             [6, 79, 3, 35, 1],
             [7, 80, 4, 36, 2]],
            [[5, 78, 2, 34, 0],
             [6, 79, 3, 35, 1],
             [7, 80, 4, 36, 2]]])
print(x.ndim)
print(x.shape)
print(x.dtype)

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

my_slice = train_images[10:100]
print(my_slice.shape)
my_slice = train_images[:, 14:, 14:]
digit = my_slice[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.reshape((6, 1)))
print(np.transpose(x))
