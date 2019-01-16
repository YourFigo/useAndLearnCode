import matplotlib.pyplot as plt
import LoadingData as ld
import numpy as np
import os
import itertools
import WritingData as wd

# method = ['SEMIFS_that','spec_that','MCFS','IG']
# mark = ['o-','x-','^-','.--']
# method = ['MRMR','fisher','SEMICHI','IG']
# mark = ['o-','x-','^-','.--']

#获得文件夹中所有非目录文件的名字，返回所有文件名的列表
def get_file_name_list(file_dir):
    for root, dirs, files in os.walk(file_dir):
        pass
    return files

#判定file_name是否是path路径中某个文件名的子串，如果是，返回文件名，返回的file_list[i]相当于给file_name加上了日期
def get_curr_file_name(path,file_name):
    file_list = get_file_name_list(path)
    for i in range(len(file_list)):
        if file_name in file_list[i]:
            return file_list[i]
            break
#获得方法为method，文件名为fileName的准确率，并存储在一个列表中返回，相当于获得了画图的Y轴
def getAccList(method,fileName,size,classifier):
    path  = 'acc' + '/' + classifier + '/' + size + '/' + method  #换分类器需要修改
    curr_file_name = get_curr_file_name(path,fileName)
    Path_file = path + '/' + curr_file_name
    _, dataMatrix = ld.import_csv(Path_file)
    accList = []
    for i in range(len(dataMatrix[:, 1])):
        accList.append(dataMatrix[i, 1])
    return accList
#获得画图的X轴
def getAcc_x(index):
    acc_x = np.array(range(0,selected_num[index], step[index]))
    return acc_x+1

if __name__ == '__main__':

    size = ['s', 'm', 'b','o']
    # classfiers_name = ['Gauss', 'GBC', 'KNN', 'RFC', 'Tree']
    classfiers_name = ['Gauss', 'GBC', 'KNN', 'RFC', 'Tree']
    method = ['IG', 'MCFS', 'reliefF', 'SEMIFS_MM', 'SPEC']
    mark = ['x--',  's--',  '^--', '.-', '*--', 'p--','*-']

    file_name = ['Hill_Valley_with_noise', 'datasmall', 'SRBCT',
                 'Leukemia1', 'DLBCL', 'Lung1', 'madelon']
    selected_num = [100, 160, 500, 500, 500, 500, 300]
    step = [3, 4, 10, 10, 10, 10, 10]

    file_name = ['biodeg', 'ForestTypes', 'heart',
                 'movement_libras', 'plrx', 'pop_failures']
    selected_num = [40, 25, 13, 90, 12, 18]
    step = [1, 1, 1, 3, 1, 1]

    # 得到分类器的组合
    classfiers_combs = [list(itertools.combinations(classfiers_name,i)) for i in range(2,6)]
    print(classfiers_combs)


    for classfiers_comb in classfiers_combs:
        for classfiers in classfiers_comb:
            #  上面两个循环就取出了一个组合
            for i in range(len(file_name)):
                acc_x = getAcc_x(i)
                plt.figure()
                # print(classfiers)
                # print(len(classfiers))
                for j in range(len(method)):
                    classfiers_num = len(classfiers)
                    image_name = file_name[i]
                    acc = 0
                    for classfier in classfiers:
                        acc_one = getAccList(method[j], file_name[i], 'o', classfier)  # 换大小数据集需要修改
                        acc_one = np.array(acc_one)
                        acc = acc + acc_one
                        image_name = image_name + '_' + classfier
                    acc = acc / classfiers_num
                    plt.plot( acc_x,acc, mark[j])

                plt.xlabel("num of features")
                plt.ylabel("acc")
                plt.legend(['IG', 'MCFS', 'reliefF', 'SEMIFS', 'SPEC'],  # 换特征选择方法需要修改
                           loc="best", fontsize='x-small')
                plt.grid(True)  # 换大小数据集需要修改
                img_path = 'image' + '/' + 'comb' + '/' + 'o' + '/' + file_name[i] + '/'
                wd.mkdir(img_path)
                plt.savefig(img_path + image_name + '.svg', format='svg')