# 20世纪70年代波士顿房价回归问题
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# 注意，用于测试数据标准化的均值和标准差都是在训练数据上计算得到的。
test_data -= mean
test_data /= std

from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    # 网络的最后一层只有一个单元，没有激活，是一个线性层。这是标量回归（标量回归
    # 是预测单一连续值的回归）的典型设置。添加激活函数将会限制输出范围。
    model.add(layers.Dense(1))
    # 编译网络用的是 mse 损失函数，即均方误差（MSE， mean squared error），预测值与
    # 目标值之差的平方。这是回归问题常用的损失函数。
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    # 平均绝对误差（MAE， mean absolute error）。它是预测值与目标值之差的绝对值。
    # 比如，如果这个问题的 MAE 等于 0.5，就表示你预测的房价与实际价格平均相差 500 美元。
    return model

# 由于数据点过少，因此采用K折交叉验证，否则验证集的结果将会不稳定。模型的验证分数等于 K 个验证分数的平均值。
#import numpy as np
#k = 4
#num_val_samples = len(train_data) // k
#num_epochs = 100
#all_scores = []
#
#for i in range(k):
#    print('processing fold #', i)
#    # 验证集 [i:i+1]
#    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#    # 训练集 [0:i]∪[i+1:n]
#    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
#                                         train_data[(i + 1) * num_val_samples:]],axis=0)
#    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
#                                            train_targets[(i + 1) * num_val_samples:]],axis=0)
#    
#    model = build_model()
#    model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=1, verbose=0)
#    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#    all_scores.append(val_mae)
#
#print(all_scores)
#print(np.mean(all_scores))

# 增加了训练轮数，为了记录模型在每轮的表现，我们需要修改训练循环，以保存每轮的验证分数记录。
import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

#计算所有轮次中的 K 折验证分数平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print('average_mae_history : ',average_mae_history)

#绘制验证分数
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 绘制验证分数（删除前 10 个数据点）
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 训练最终模型
model = build_model()
model.fit(train_data, train_targets,
epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('test_mse_score : {} \n test_mae_score : {}'.format(test_mse_score,test_mae_score))