import numpy.matlib
import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import rbf_kernel
from numpy import linalg as LA


def spec(X, **kwargs):
    """
    This function implements the SPEC feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        style: {int}
            style == -1, the first feature ranking function, use all eigenvalues
            style == 0, the second feature ranking function, use all except the 1st eigenvalue
            style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
        W: {sparse matrix}, shape (n_samples, n_samples}
            input affinity matrix

    Output
    ------
    w_fea: {numpy array}, shape (n_features,)
        SPEC feature score for each feature

    Reference
    ---------
    Zhao, Zheng and Liu, Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
    """

    if 'style' not in kwargs:
        kwargs['style'] = 0
    if 'W' not in kwargs:
        kwargs['W'] = rbf_kernel(X, gamma=1)   #高斯核函数

    style = kwargs['style']
    W = kwargs['W']
    if type(W) is numpy.ndarray:
        W = csc_matrix(W)    #Compressed Sparse Column Format (CSC) 压缩矩阵

    n_samples, n_features = X.shape

    # build the degree matrix
    X_sum = np.array(W.sum(axis=1))      #关于axis：轴用来为超过一维的数组定义的属性，二维数据拥有两个轴：第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸。
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        D[i, i] = X_sum[i]

    # build the laplacian matrix
    L = D - W
    d1 = np.power(np.array(W.sum(axis=1)), -0.5)
    d1[np.isinf(d1)] = 0
    d2 = np.power(np.array(W.sum(axis=1)), 0.5)
    v = np.dot(np.diag(d2[:, 0]), np.ones(n_samples))
    v = v/LA.norm(v)

    # build the normalized laplacian matrix
    L_hat = (np.matlib.repmat(d1, 1, n_samples)) * np.array(L) * np.matlib.repmat(np.transpose(d1), n_samples, 1)

    # calculate and construct spectral information
    s, U = np.linalg.eigh(L_hat)   #计算拉普拉斯矩阵的特征值s和特征向量U
    s = np.flipud(s)
    U = np.fliplr(U)

    # begin to select features   w_fea   每个特征的feature score
    w_fea = np.ones(n_features)*1000     #   [1000,1000,1000,...,1000]   n_features个元素

    for i in range(n_features):
        f = X[:, i]
        F_hat = np.dot(np.diag(d2[:, 0]), f)#dot()矩阵乘法 diag()形成对角矩阵
        l = LA.norm(F_hat)      #norm()求矩阵范数，默认为Frobenius范数，计算方式为矩阵元素的绝对值的平方和再开方
        if l < 100*np.spacing(1):      #spacing(1)距离1最近的浮点数
            w_fea[i] = 1000
            continue
        else:
            F_hat = F_hat/l
        a = np.array(np.dot(np.transpose(F_hat), U))
        a = np.multiply(a, a)
        a = np.transpose(a)

        # use f'Lf formulation
        if style == -1:
            w_fea[i] = np.sum(a * s)
        # using all eigenvalues except the 1st
        elif style == 0:
            a1 = a[0:n_samples-1]
            w_fea[i] = np.sum(a1 * s[0:n_samples-1])/(1-np.power(np.dot(np.transpose(F_hat), v), 2))
        # use first k except the 1st
        else:
            a1 = a[n_samples-style:n_samples-1]
            w_fea[i] = np.sum(a1 * (2-s[n_samples-style: n_samples-1]))

    if style != -1 and style != 0:
        w_fea[w_fea == 1000] = -1000

    return w_fea


def feature_ranking(score, **kwargs):
    if 'style' not in kwargs:
        kwargs['style'] = 0
    style = kwargs['style']

    # if style = -1 or 0, ranking features in descending order, the higher the score, the more important the feature is
    if style == -1 or style == 0:
        idx = np.argsort(score, 0)
        return idx[::-1]
    # if style != -1 and 0, ranking features in ascending order, the lower the score, the more important the feature is
    elif style != -1 and style != 0:
        idx = np.argsort(score, 0)
        return idx

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

        feature_score = spec(X_train)
        wd.out_feature_score('SPEC', feature_score, classifier, file_name, count, size)
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
    print(selected_num,'\t', 'SPEC ', '\t', acc)
    return acc