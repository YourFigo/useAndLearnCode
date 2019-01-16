import LoadingData as ld
import numpy as np
import pandas as pd
import datetime
import WritingData as wd
import semi_chi
import IG
import f_score
import fisher_score
import reliefF
import chi_square
import MCFS
import SPEC

#文件名称、选多少个特征、隔多少个特征取一次、大数据集还是小数据集
def outResult(FS,file_name,selected_num,step = 10,size = 's',classifier = 'Gauss',k = 10):
    open_file = 'dataset/' + file_name + '.csv'
    if size == 's':
        open_file = 'dataset/data_small/' + file_name + '.csv'
    elif size == 'm':
        open_file = 'dataset/data_middle/' + file_name + '.csv'
    elif size == 'b':
        open_file = 'dataset/data_big/' + file_name + '.csv'
    elif size == 'o':
        open_file = 'dataset/other/' + file_name + '.csv'
    data = ld.openfile(open_file)
    X,y = ld.splitData(data)

    acc = []

    count_i = []

    # score_chi = []
    print('此数据集共取 ' + str(selected_num/step) + ' 次特征')
    for i in range(1, selected_num + 1, step):
        startTime = datetime.datetime.now()
        print('********取第  ' + str(i) +'  个特征值*********')
        count_i.append(i)

        if FS == 'SEMIFS_MM':
            acc.append(semi_chi.getACC(size,file_name,X, y, i,classifier,k))
        elif FS == 'CHI':
            acc.append(chi_square.getACC(size, file_name, X, y, i, classifier, k))
        elif FS == 'f_score':
            acc.append(f_score.getACC(size, file_name, X, y, i, classifier, k))
        elif FS == 'fisher':
            acc.append(fisher_score.getACC(size, file_name, X, y, i, classifier, k))
        elif FS == 'IG':
            acc.append(IG.getACC(size, file_name, X, y, i, classifier, k))
        elif FS == 'MCFS':
            acc.append(MCFS.getACC(size, file_name, X, y, i, classifier, k))
        elif FS == 'reliefF':
            acc.append(reliefF.getACC(size, file_name, X, y, i, classifier, k))
        elif FS == 'SPEC':
            acc.append(SPEC.getACC(size, file_name, X, y, i, classifier, k))

        endTime = datetime.datetime.now()
        print('取第 ' + str(i) + ' 个特征耗时：', end='')
        ld.runtime(startTime, endTime)

    wd.out_acc(FS, acc, classifier, file_name, size)

if __name__ == '__main__':
    startTime = datetime.datetime.now()

    file_name = ['SCADI', 'Cryotherapy', 'soybean_small', 'zoo',
                 'lung_cancer', 'lymphography', 'glass', 'breast-cancer-wisconsin', 'SPECTF', 'dermatology',
                 'Z-Alizadeh_sani_dataset', 'ionosphere', 'sonar', 'Urban_land_cover', 'movement_libras_1',
                 'movement_libras_5', 'movement_libras_8', 'movement_libras_9', 'movement_libras_10']
    selected_num = [150, 6, 35, 16, 55, 18, 9, 9, 44, 34, 55, 32, 60, 147, 90, 90, 90, 90, 90]
    step = [5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 4, 3, 3, 3, 3, 3]

    for i in range(len(file_name)):
        try:
            print('---------**** runing : ', '第 ', i + 1, ' 个数据集 *******---------')
            outResult('SEMIFS_MM', file_name[i], selected_num[i], step[i], 's', 'GBC')
            outResult('CHI', file_name[i], selected_num[i], step[i], 's', 'GBC')
            outResult('f_score', file_name[i], selected_num[i], step[i], 's', 'GBC')
            outResult('fisher', file_name[i], selected_num[i], step[i], 's', 'GBC')
            outResult('IG', file_name[i], selected_num[i], step[i], 's', 'GBC')
            outResult('MCFS', file_name[i], selected_num[i], step[i], 's', 'GBC')
            outResult('reliefF', file_name[i], selected_num[i], step[i], 's', 'GBC')
            outResult('SPEC', file_name[i], selected_num[i], step[i], 's', 'GBC')
        except:
            print('this dataset arise error')
            continue

    # file_name = ['biodeg', 'ForestTypes', 'heart', 'Hill_Valley_without_noise',
    #              'movement_libras', 'plrx', 'pop_failures']
    # selected_num = [40, 25, 13, 100, 90, 12, 18]
    # step = [1, 1, 1, 3, 3, 1, 1]
    #
    # for i in range(len(file_name)):
    #     try:
    #         print('---------**** runing : ', '第 ', i + 1, ' 个数据集 *******---------')
    #         outResult('SEMIFS_MM', file_name[i], selected_num[i], step[i], 'o', 'GBC')
    #         outResult('CHI', file_name[i], selected_num[i], step[i], 'o', 'GBC')
    #         outResult('f_score', file_name[i], selected_num[i], step[i], 'o', 'GBC')
    #         outResult('fisher', file_name[i], selected_num[i], step[i], 'o', 'GBC')
    #         outResult('IG', file_name[i], selected_num[i], step[i], 'o', 'GBC')
    #         outResult('MCFS', file_name[i], selected_num[i], step[i], 'o', 'GBC')
    #         outResult('reliefF', file_name[i], selected_num[i], step[i], 'o', 'GBC')
    #         outResult('SPEC', file_name[i], selected_num[i], step[i], 'o', 'GBC')
    #     except:
    #         print('this dataset arise error')
    #         continue

    # file_name = [ 'Hill_Valley_with_noise', 'datasmall', 'SRBCT',
    #              'Leukemia1', 'DLBCL', 'Lung1', 'madelon']
    # selected_num = [100, 160, 500, 500, 500, 500, 300]
    # step = [ 3, 4, 10, 10, 10, 10, 10]
    # for i in range(len(file_name)):
    #     try:
    #         print('---------**** runing : ', '第 ', i + 1, ' 个数据集 *******---------')
    #         outResult('SEMIFS_MM',file_name[i], selected_num[i], step[i], 'm', 'GBC')
    #         outResult('CHI', file_name[i], selected_num[i], step[i], 'm', 'GBC')
    #         outResult('f_score', file_name[i], selected_num[i], step[i], 'm', 'GBC')
    #         outResult('fisher', file_name[i], selected_num[i], step[i], 'm', 'GBC')
    #         outResult('IG', file_name[i], selected_num[i], step[i], 'm', 'GBC')
    #         outResult('MCFS', file_name[i], selected_num[i], step[i], 'm', 'GBC')
    #         outResult('reliefF', file_name[i], selected_num[i], step[i], 'm', 'GBC')
    #         outResult('SPEC', file_name[i], selected_num[i], step[i], 'm', 'GBC')
    #     except:
    #         print('this dataset arise error')
    #         continue

    endTime = datetime.datetime.now()
    print('总耗时：', end='')
    ld.runtime(startTime, endTime)
