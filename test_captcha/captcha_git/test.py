import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
data_dir = 'D:/3_other_code/data/captcha_git/images_test'

width = 160
height = 60
max_captcha = 4
batch_size = 64
num_numbers = len(number)


def get_train_data(data_dir=data_dir):
    simples = {}
    for file_name in os.listdir(data_dir):
        captcha = file_name.split('.')[0]
        simples[data_dir + '/' + file_name] = captcha
    return simples


simples = get_train_data(data_dir)
file_simples = list(simples.keys())
num_simples = len(simples)


def test(input_, label_):
    saver = tf.train.import_meta_graph(model_path)
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('input:0')
    predict_max_idx = graph.get_tensor_by_name('predict_max_idx:0')
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        predict = sess.run(predict_max_idx, feed_dict={inputs: [input_]})
        result = (predict[0] == label_)
        print("test:{}  label:{}  TF:{}".format(predict[0], label_, result))
    return result


if __name__ == '__main__':

    num_epoch = 20000
    model_dir = 'D:/3_other_code/data/captcha_git/model_' + str(num_epoch)
    model_path = 'D:/3_other_code/data/captcha_git/model_' + \
                 str(num_epoch) + '/output.model-' + str(num_epoch - 100) + '.meta'

    count = 0
    numtrue = 0
    for i in range(num_simples):
        input_ = np.float32(cv2.imread(file_simples[i], 0)).flatten() / 255
        label_ = [ord(captcha) - 48 for captcha in simples[file_simples[i]]]
        results = test(input_, label_)
        for result in results:
            if result == True:
                numtrue = numtrue + 1
        t = np.array([True, True, True, True])
        if np.array_equal(results,t):
            count = count + 1
            # print(file_simples[i])

    print('正确率：%.2f%%(%d/%d)' % (count * 100 / num_simples, count, num_simples))
    print('单个数字正确率：%.2f%%(%d/%d)' % ((numtrue * 100) / (num_simples * 4), numtrue, num_simples * 4))

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    with open(model_dir + '/test_acc.txt','w') as f:
        f.write('正确率：%.2f%%(%d/%d)' % (count * 100 / num_simples, count, num_simples))
        f.write('\n')
        f.write('单个数字正确率：%.2f%%(%d/%d)' % ((numtrue * 100) / (num_simples * 4), numtrue, num_simples * 4))
