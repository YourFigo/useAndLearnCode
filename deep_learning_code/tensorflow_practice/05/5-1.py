import numpy as np
import cv2
img = np.mat(np.zeros((300,300)))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j] = i+j
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
cv2.imshow("test",img)
cv2.waitKey(0)
