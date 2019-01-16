import numpy as np

file_name = ['biodeg', 'ForestTypes', 'heart',
                 'movement_libras', 'plrx', 'pop_failures']
selected_num = [40, 25, 13, 90, 12, 18]
step = [1, 1, 1, 3, 3, 1, 1]



def getAcc_x(index):
    acc_x = np.array(range(0,selected_num[index], step[index]))
    return acc_x+1

for i in range(len(file_name)):
    acc_x = getAcc_x(i)
    print(len(acc_x))
    print(acc_x)