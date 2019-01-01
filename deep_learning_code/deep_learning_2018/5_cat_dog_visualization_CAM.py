# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:28:05 2018

@author: Figo
"""

from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
img_path = 'D:/3_other_code/data/kaggle/cats_and_dogs_small/female-elephant-341983_1280.jpg'
#大 小 为 224×224 的 Python图像库（PIL， Python imaging library）图像
img = image.load_img(img_path, target_size=(224, 224))
#形 状 为 (224, 224, 3)
x = image.img_to_array(img)
#添加一个维度，将数组转换为(1, 224, 224, 3) 形状的批量
x = np.expand_dims(x, axis=0)
#对批量进行预处理（按通道进行颜色标准化）
x = preprocess_input(x)

# 现在你可以在图像上运行预训练的 VGG16 网络，并将其预测向量解码为人类可读的格式。
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
print(np.argmax(preds[0]))

african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('D:/3_other_code/data/kaggle/cats_and_dogs_small/elephant_cam.jpg', superimposed_img)