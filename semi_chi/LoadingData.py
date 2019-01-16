import csv as csv
import numpy as np
from itertools import islice

#flag 和 has_title_line 这两个参数多余了
def import_data(file_name):
    # open file given the file name and
    # return a 2-dimension list
    file_data = []
    row_count = 0
    #print('Loading file: ' + file_name)
    try:
        csv_file = open(file_name, 'r')
        lines = csv.reader(csv_file)
        for line in lines:
            file_data.append(line)
            row_count += 1
        csv_file.close()
        # if flag == 1:  #标记为1，不读取第一行
        #     for line in islice(lines, flag, None):
        #         file_data.append(line)
        #         row_count += 1
        col_count = file_data[0].__len__()
        print(file_name + ' 导入成功  ！！')
        print('数据大小: ' + str(row_count) + ' rows, ' + str(col_count) + ' cols')
    except Exception as e:
        if not file_data == []:
            print("Current Data: row " + str(row_count))
            print(file_data[-1])
        print(e)
    else:
        return file_data

def import_matrix(file_name):
    # import the shuffled data as matrix
    # attention, feature_set and label_set shall imported separately
    data = import_data(file_name)
    #print('data',data)
    # if has_title_line:
    # title_names = np.array(data[0])
    # data_matrix = np.mat(data[1:], dtype=np.float64)
    # print('import matrix successfully,have title')
    # return title_names, data_matrix
    # else:
    data_matrix = np.mat(data[0:], dtype=np.float64)
    return None,data_matrix
        # print('import matrix successfully,no title')

def import_csv(file_name):
    # import the shuffled data as matrix
    # attention, feature_set and label_set shall imported separately
    data = import_data(file_name)
    #print('data',data)
    # if has_title_line:
    # title_names = np.array(data[0])
    # data_matrix = np.mat(data[1:], dtype=np.float64)
    # print('import matrix successfully,have title')
    # return title_names, data_matrix
    # else:
    data_matrix = np.mat(data[1:], dtype=np.float64)
    return None,data_matrix
        # print('import matrix successfully,no title')

def openfile(path):
    with open(path, "rt")as data:
        reader = csv.reader(data)
        my_data = [row for row in reader]
    dataset = np.array(my_data)
    return dataset

def splitData(dataSet):
    character=[]
    label=[]
    for i in range(len(dataSet)):
        character.append([float(tk) for tk in dataSet[i][:-1]])
        label.append(dataSet[i][-1])
    return np.array(character),np.array(label)


# 通过训练集得到最优子集后，然后通过最优子集处理测试集
def create_test_matrix(file_name_test, sort_index, has_title_line=False):
    # 加载测试文件
    title_names_test, data_matrix_test = import_matrix(file_name_test, has_title_line)
    nrow, ncol = data_matrix_test.shape
    no_class_col = ncol - 1
    # 将测试集按照训练集的特征排序进行处理
    # 先处理特征
    out_matrix = np.zeros(shape=(nrow, ncol))
    for col in range(no_class_col):
        for row in range(nrow):
            out_matrix[row, col] = data_matrix_test[row, sort_index[col]]
    # 再将类别标签直接赋值过来
    out_matrix[:, ncol - 1] = (data_matrix_test[:, ncol - 1]).T
    return out_matrix


#运行时间
def runtime(startTime,endTime):
    runTime = (endTime - startTime).seconds
    print(runTime,' seconds')


if __name__ == '__main__':
    file_name = r'dataset\spect.csv'
    file_data_0 = import_data(file_name)
    file_data_1 = import_data(file_name,1)
    # print('file_data')
    # print(file_data)
    # title_names,data_matrix = import_matrix(file_name,has_title_line=False)
    # col = data_matrix.shape[1]
    # print('field_names')
    # print(title_names)
    # print('data_matrix')
    # print(data_matrix)
    # print('data_matrix[0]:',data_matrix[0][:])
    # runtime()