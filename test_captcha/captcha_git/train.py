import numpy as np
import tensorflow as tf
import cv2
import os
import random
import matplotlib.pyplot as plt
import time

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num_numbers = len(number)
# 每张验证码图片中的数字个数
max_captcha = 4

# 训练图片位置
data_dir = 'D:/3_other_code/data/captcha_git/images'

# 每张图片的宽度、高度
width = 160
height = 60

# batch 大小
batch_size = 64

# 用于获取训练样本，返回一个存储着图片路径和标签的字典
def get_train_data(data_dir=data_dir):
    simples = {}
    for file_name in os.listdir(data_dir):
        captcha = file_name.split('.')[0]
        simples[data_dir + '/' + file_name] = captcha
    return simples

# 获得训练样本
simples = get_train_data(data_dir)
# 获得训练样本中的图片路径
file_simples = list(simples.keys())
num_simples = len(simples)

# 将每张图片的label转换为40维向量存储,一共4个数字，每10维存一个数字。假如数字为5，则10位当中的第5位是1，其他位是0
#例如，如果验证码是 ‘0296’ ，则对应的标签是
# [1 0 0 0 0 0 0 0 0 0
#  0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1
#  0 0 0 0 0 0 1 0 0 0]
def text2vec(text):
    # ord(0) = 48
    return [0 if ord(i) - 48 != j else 1 for i in text for j in range(num_numbers)]

def text2vec1(text):
    vec = []
    for i in text:
        for j in range(num_numbers):
            x = ord(i)
            if ord(i) - 48 != j:
                vec.append(0)
            else:
                vec.append(1)
    return vec

# 生成一个batch用于训练
def get_next_batch():
    # batch_x:[64,160*6] = [64,9600] 用于存放每批64张图片，每张图片以每个像素存放
    batch_x = np.zeros([batch_size, width * height])
    # batch_y:[64,10*4] = [64,40]
    batch_y = np.zeros([batch_size, num_numbers * max_captcha])

    for i in range(batch_size):
        file_name = file_simples[random.randint(0, num_simples - 1)]
        # cv2.imread()接口读图像，读进来直接是BGR格式，数据格式在 0~255，通道格式为(W,H,C)
        # -1 =< flag <= 3； >0 Return a 3-channel color image ；=0 Return a grayscale image；<0 Return the loaded image as is (with alpha channel)
        # flatten() 将np矩阵折叠为一维形式
        batch_x[i, :] = np.float32(cv2.imread(file_name, 0)).flatten() / 255
        batch_y[i, :] = text2vec(simples[file_name])
        # batch_y[i, :] = text2vec1(simples[file_name])
    return batch_x, batch_y


# 构造网络并训练cnn
def create_train_cnn(epoch = 5000):

    #####################定义用于构造网络的一些初始化函数#######################
    # 初始化权重
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 初始化偏置
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 设置卷积层
    def conv2d(x, W):
        # tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,data_format=None,name=None)
        # input：做卷积输入的图像，是一个4维的Tensor，[batch_size, in_height, in_width, in_channels]
        # filter：卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape
        # strides：这是一个一维的向量，长度为4，对应的是在input的4个维度上的步长
        # padding：只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式，SAME代表卷积核可以停留图像边缘，VALID表示不能
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 设置池化层
    def max_pool_2x2(x):
        # tf.nn.max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)
        # value：需要池化的输入，依然是[batch, height, width, channels]这样的shape
        # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        # strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
        # padding：和卷积类似，可以取'VALID' 或者'SAME'
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ##############################构造网络######################################
    # 输入层
    # tf.placeholder():函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
    x = tf.placeholder(tf.float32, [None, width * height], name='input')
    y_ = tf.placeholder(tf.float32, [None, num_numbers * max_captcha])
    x_image = tf.reshape(x, [-1, height, width, 1])

    # dropout,防止过拟合
    # 请注意 keep_prob 的 name，在测试model时会用到它
    # keep_prob = tf.placeholder(tf.float32, name='keep-prob')

    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

    # 第三层卷积
    W_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # h_pool3 = tf.nn.dropout(h_pool3, keep_prob)

    # 全连接层
    W_fc1 = weight_variable([8 * 20 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 20 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    # h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    W_fc2 = weight_variable([1024, num_numbers * max_captcha])
    b_fc2 = bias_variable([num_numbers * max_captcha])
    output = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=output))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

    predict = tf.reshape(output, [-1, max_captcha, num_numbers])
    labels = tf.reshape(y_, [-1, max_captcha, num_numbers])
    correct_prediction = tf.equal(tf.argmax(predict, 2, name='predict_max_idx'), tf.argmax(labels, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ##############################训练网络######################################
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    stepsList = []
    accsList = []
    for i in range(epoch):
        batch_x, batch_y = get_next_batch()
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            print("step %d, training accuracy %g " % (i, train_accuracy))
            if train_accuracy > 0.99:
                saver.save(sess, model_path, global_step=i)
            stepsList.append(i)
            accsList.append(train_accuracy)
        train_step.run(feed_dict={x: batch_x, y_: batch_y})
    return stepsList,accsList


if __name__ == '__main__':

    epoch = 20000
    # 训练好的模型的存储位置
    model_dir = "D:/3_other_code/data/captcha_git/model_" + str(epoch)
    model_path = "D:/3_other_code/data/captcha_git/model_" + str(epoch) + "/output.model"
    plt_dir = "D:/3_other_code/data/captcha_git/model_" + str(epoch) + "/plt_train_acc.png"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    stepsList, accsList = create_train_cnn(epoch)

    plt.plot(stepsList, accsList)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.savefig(plt_dir)
    print('Training finished')