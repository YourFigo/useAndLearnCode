import numpy as np
import mutual_information as mi

def IGfs(X_train,y_train):

    nrow,ncol = X_train.shape
    col_class = np.array(y_train)
    # 监督评分（SU）
    feature_score = np.zeros(ncol)

    for col in range(ncol):
        col_curr = np.array(X_train[:,col]).transpose()
        feature_score[col] = mi.information_gain(col_curr, col_class)

    return feature_score

def feature_ranking(F):
    idx = np.argsort(F)
    return idx[::-1]

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import WritingData as wd
from sklearn import tree
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
    for train_index, test_index in skf.split(X, y):
        count = count + 1
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        n_row_train,n_col_train = X_train.shape
        n_row_test,n_col_test = X_test.shape
        feature_train = np.zeros((n_row_train,n_col_train))
        feature_test = np.zeros((n_row_test,n_col_test))

        feature_score = IGfs(X_train,y_train)
        wd.out_feature_score('IG', feature_score, classifier, file_name, count, size)
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
    print(selected_num,'\t', 'IG', '\t', acc)
    return acc