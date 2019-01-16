import pandas as pd
import numpy as np
import os
import csv
import datetime

#写入csv文件
def insert_csv(data_matrix,write_fileName,k):
    nrow,ncol = data_matrix.shape
    #取data_matrix得前k-1个特征，并且将最后一个列（标签）赋值给data_matrix_Top的第k个
    data_matrix_Top = np.zeros((nrow,k))
    for i in range(k):
        data_matrix_Top[:,i] = data_matrix[:, i]
    data_matrix_Top[:, k - 1] = (data_matrix[:, ncol - 1]).T

    try:
        pd_data = pd.DataFrame(data_matrix_Top)
        pd_data.to_csv(write_fileName)
        print('out csv successful')
    except Exception as e:
        print(e)
        pass

#创建文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if folder:
        pass
        # print(path,'  already exist !')
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("create new folder: ",path)

def out_feature_score(FS,feature_score, classifier, file_name, count,size):
    feature_score = np.array(feature_score).transpose()
    try:
        out_matrix = np.c_[feature_score]
        pd_data = pd.DataFrame(out_matrix)
        outPath = 'feature_score' + '/' + classifier + '/'+ size + '/' + FS
        mkdir(outPath)
        pd_data.to_csv(outPath + '/' + file_name + '_' + str(count) + '.csv')
    except Exception as e:
        print('feature_score result fail   ', outPath)
        print(e)

def out_acc(FS, acc, classifier, file_name, size):
    nowTime = datetime.datetime.now()
    nowTime = nowTime.strftime('%Y-%m-%d-%H-%M-%S')
    acc = np.array(acc).transpose()
    try:
        out_matrix = np.c_[acc]
        pd_data = pd.DataFrame(out_matrix)
        outPath = 'acc' + '/' + classifier + '/' + size + '/' + FS
        mkdir(outPath)
        pd_data.to_csv(outPath + '/' + file_name + '_' + str(nowTime) + '.csv')
    except Exception as e:
        print(e)

#写入txt文件
def insert_result(sort_index,feature_score,write_resultTxt):
    sort_index = np.mat(sort_index).T
    feature_score = np.mat(feature_score).T
    try:
        out_matrix = np.c_[sort_index,feature_score]
        pd_data = pd.DataFrame(out_matrix)
        pd_data.to_csv(write_resultTxt)
        print('out result successful')
    except Exception as e:
        print(e)
        pass

def insert_dir(open_file_name,num_select):

    mkdir_path1 = 'resultDataset/' + open_file_name + '/' + str(num_select)
    mkdir_path2 = 'resultDataInfo/' + open_file_name + '/' + str(num_select)
    mkdir(mkdir_path1)
    mkdir(mkdir_path2)

#删除csv指定列
def del_cvs_col(fname, newfname, idxs, delimiter=' '):
    with open(fname) as csvin, open(newfname, 'w',newline='') as csvout:
        reader = csv.reader(csvin, delimiter=delimiter)
        writer = csv.writer(csvout, delimiter=delimiter,dialect='excel')
        rows = (tuple(item for idx, item in enumerate(row) if idx not in idxs) for row in reader)
        writer.writerows(rows)
    os.remove(fname)



if __name__ == '__main__':
    # insert_result([1,2,3,4,5],[0,0,0,0,0],'text.txt')
    fileName1 = 'resultDataset/isolet5/dataset_out_se_isolet5_0_1.csv'
    fileName2 = 'resultDataset/isolet5/zzdataset_out_se_isolet5_0_1.csv'
    del_cvs_col(fileName1,fileName2,[0],',')