from captcha.image import ImageCaptcha
import shutil
import sys
import numpy as np
from PIL import Image
import random
import cv2
import os
import time

CAPTCHA_IMAGE_PATH = 'D:/3_other_code/GitCode/useAndLearnCode/test_captcha/captcha_git/images11/'
TEST_IMAGE_PATH = 'D:/3_other_code/GitCode/useAndLearnCode/test_captcha/captcha_git/images_test11/'
TEST_IMAGE_NUMBER = 100

# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
 
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z']
# ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#             'V', 'W', 'X', 'Y', 'Z']
 
# 验证码长度为4个字符
# 这个函数是随机生成的，所以最后可能有重复的，因此生成不可能超过10000个
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text
 
 
# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
 
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
 
    captcha = image.generate(captcha_text)
 
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

def generate_image():
    for i in range(0,10000,1):
        text, image = gen_captcha_text_and_image()
        fullPath = os.path.join(CAPTCHA_IMAGE_PATH, text + ".jpg")
        cv2.imwrite(fullPath, image)
        if i % 1000 == 0:
            time.sleep(2)
        sys.stdout.write("\rCreating %d/%d" % (i, 10000))
        sys.stdout.flush()
    print ("\nDone!")

def prepare_test_set():
    fileNameList = []
    for filePath in os.listdir(CAPTCHA_IMAGE_PATH):
        captcha_name = filePath.split('/')[-1]
        fileNameList.append(captcha_name)
    random.seed(time.time())
    random.shuffle(fileNameList)
    for i in range(TEST_IMAGE_NUMBER):
        name = fileNameList[i]
        shutil.copy(CAPTCHA_IMAGE_PATH + name, TEST_IMAGE_PATH + name)

if __name__ == '__main__':

    generate_image()
    prepare_test_set()
    sys.stdout.write("\nFinished")
    sys.stdout.flush()