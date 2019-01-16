import scipy
import numpy as np
from sklearn import linear_model
from skfeature.utility.construct_W import construct_W


def mcfs(X, n_selected_features, **kwargs):
    """
    This function implements unsupervised feature selection for multi-cluster data.

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_selected_features: {int}
        number of features to select
    kwargs: {dictionary}
        W: {sparse matrix}, shape (n_samples, n_samples)
            affinity matrix
        n_clusters: {int}
            number of clusters (default is 5)

    Output
    ------
    W: {numpy array}, shape(n_features, n_clusters)
        feature weight matrix

    Reference
    ---------
    Cai, Deng et al. "Unsupervised Feature Selection for Multi-Cluster Data." KDD 2010.
    """

    # use the default affinity matrix
    if 'W' not in kwargs:
        W = construct_W(X)
    else:
        W = kwargs['W']
    # default number of clusters is 5
    if 'n_clusters' not in kwargs:
        n_clusters = 5
    else:
        n_clusters = kwargs['n_clusters']

    # solve the generalized eigen-decomposition problem and get the top K
    # eigen-vectors with respect to the smallest eigenvalues
    W = W.toarray()
    W = (W + W.T) / 2
    W_norm = np.diag(np.sqrt(1 / W.sum(1)))
    W = np.dot(W_norm, np.dot(W, W_norm))
    WT = W.T
    W[W < WT] = WT[W < WT]
    eigen_value, ul = scipy.linalg.eigh(a=W)
    Y = np.dot(W_norm, ul[:, -1*n_clusters-1:-1])

    # solve K L1-regularized regression problem using LARs algorithm with cardinality constraint being d
    n_sample, n_feature = X.shape
    W = np.zeros((n_feature, n_clusters))
    for i in range(n_clusters):
        clf = linear_model.Lars(n_nonzero_coefs=n_selected_features)
        clf.fit(X, Y[:, i])
        W[:, i] = clf.coef_
    return W


def feature_ranking(W):
    """
    This function computes MCFS score and ranking features according to feature weights matrix W
    """
    mcfs_score = W.max(1)
    idx = np.argsort(mcfs_score, 0)
    idx = idx[::-1]
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

        feature_score = mcfs(X_train,n_selected_features=selected_num)
        wd.out_feature_score('MCFS', feature_score, classifier, file_name, count, size)
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
    print(selected_num,'\t', 'MCFS', '\t', acc)
    return acc