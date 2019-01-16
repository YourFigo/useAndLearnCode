import chi_square
import SPEC
import WritingData as wd
import numpy as np
from sklearn import preprocessing

def SEMICHI(X_train, y_train):
    F1 = chi_square.chi_square(X_train, y_train)
    F1 = preprocessing.minmax_scale(F1, feature_range=(0, 1), axis=0)
    F2 = SPEC.spec(X_train)
    F2 = preprocessing.minmax_scale(F2, feature_range=(0, 1), axis=0)
    F = F1 + F2
    return F

def feature_ranking(F):
    idx = np.argsort(F)
    return idx[::-1]


from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BaseDiscreteNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def getACC(size,file_name,X, y, selected_num,classifier = 'Gauss',k = 10):
    skf = StratifiedKFold(n_splits=k)
    if classifier == 'Gauss':
        clf = GaussianNB()
    elif classifier == 'KNN':
        clf = KNeighborsClassifier()
    elif classifier == 'SVM':
        clf = SVC()
    elif classifier == 'Tree':
        clf = tree.DecisionTreeClassifier()
    elif classifier == 'MNB':
        clf = MultinomialNB()
    elif classifier == 'BDNB':
        clf = BaseDiscreteNB()
    elif classifier == 'RFC':
        clf = RandomForestClassifier()
    elif classifier == 'GBC':
        clf = GradientBoostingClassifier()
    acc = 0.0

    X = preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0)

    count = 0
    for train_index,test_index in skf.split(X,y):
        count = count + 1
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        n_row_train,n_col_train = X_train.shape
        n_row_test,n_col_test = X_test.shape
        feature_train = np.zeros((n_row_train,n_col_train))
        feature_test = np.zeros((n_row_test,n_col_test))

        feature_score = SEMICHI(X_train,y_train)
        wd.out_feature_score('SEMIFS_MM',feature_score, classifier, file_name, count,size)
        sort_index = feature_ranking(feature_score)

        for col in range(selected_num):
            for row in range(n_row_train):
                feature_train[row, col] = X_train[row, sort_index[col]]
        for col in range(selected_num):
            for row in range(n_row_test):
                feature_test[row, col] = X_test[row, sort_index[col]]

        clf.fit(feature_train, y_train)
        score = clf.score(feature_test,y_test) * 100
        acc += score
    acc = acc / k
    print(selected_num,'\t', 'SEMIFS_MM', '\t', acc)
    return acc

if __name__ == '__main__':
    import test_Gauss

    file_name = ['SCADI','Cryotherapy', 'soybean_small', 'zoo',
                 'lung_cancer', 'lymphography', 'glass', 'breast-cancer-wisconsin', 'SPECTF', 'dermatology',
                 'Z-Alizadeh_sani_dataset', 'ionosphere', 'sonar', 'Urban_land_cover']
    selected_num = [150,6,   35, 16,55,18,9,9,44,34,55, 32, 60, 147]
    step = [5,1,  1,  1,2, 1, 1,1,2, 1,2, 1, 2, 4]

    for i in range(len(file_name)):
        try:
            print('---------**** runing : ', '第 ', i + 1, ' 个数据集 *******---------')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 's', 'Gauss')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 's', 'GBC')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 's', 'Tree')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 's', 'KNN')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 's', 'RFC')
        except:
            print('this dataset arise error')
            continue
    file_name = [ 'Hill_Valley_with_noise', 'datasmall', 'SRBCT',
                     'Leukemia1', 'DLBCL', 'Lung1', 'madelon']
    selected_num = [100, 160, 500, 500, 500, 500, 300]
    step = [ 3, 4, 10, 10, 10, 10, 10]
    for i in range(len(file_name)):
        try:
            print('---------**** runing : ', '第 ', i + 1, ' 个数据集 *******---------')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 'm', 'Gauss')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 'm', 'GBC')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 'm', 'Tree')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 'm', 'KNN')
            test_Gauss.outResult('SEMIFS_DV', file_name[i], selected_num[i], step[i], 'm', 'RFC')
        except:
            print('this dataset arise error')
            continue