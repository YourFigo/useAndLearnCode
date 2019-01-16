from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import semi_chi

def getACC(X, y, selected_num,classifier = 'Gauss',k = 10):
    skf = StratifiedKFold(n_splits=k)
    if classifier == 'Gauss':
        clf = GaussianNB()
    elif classifier == 'KNN':
        clf = KNeighborsClassifier()
    elif classifier == 'SVM':
        clf = SVC()
    acc = 0.0

    X = preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0)

    for train_index,test_index in skf.split(X,y):
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        n_row_train,n_col_train = X_train.shape
        n_row_test,n_col_test = X_test.shape
        feature_train = np.zeros((n_row_train,n_col_train))
        feature_test = np.zeros((n_row_test,n_col_test))

        sort_score = semi_chi.SEMICHI(X_train,y_train)
        sort_index = semi_chi.feature_ranking(sort_score)

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
    print(selected_num,'\t', fs_type, '\t', acc)
    return acc