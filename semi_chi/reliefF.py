import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def reliefF(X, y, **kwargs):
    """
    This function implements the reliefF feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        parameters of reliefF:
        k: {int}
            choices for the number of neighbors (default k = 5)

    Output
    ------
    score: {numpy array}, shape (n_features,)
        reliefF score for each feature

    Reference
    ---------
    Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
    """

    if "k" not in kwargs.keys():
        k = 5
    else:
        k = kwargs["k"]
    n_samples, n_features = X.shape

    # calculate pairwise distances between instances
    distance = pairwise_distances(X, metric='manhattan')

    score = np.zeros(n_features)

    # the number of sampled instances is equal to the number of total instances
    for idx in range(n_samples):
        near_hit = []
        near_miss = dict()

        self_fea = X[idx, :]
        c = np.unique(y).tolist()

        stop_dict = dict()
        for label in c:
            stop_dict[label] = 0
        del c[c.index(y[idx])]

        p_dict = dict()
        p_label_idx = float(len(y[y == y[idx]]))/float(n_samples)

        for label in c:
            p_label_c = float(len(y[y == label]))/float(n_samples)
            p_dict[label] = p_label_c/(1-p_label_idx)
            near_miss[label] = []

        distance_sort = []
        distance[idx, idx] = np.max(distance[idx, :])

        for i in range(n_samples):
            distance_sort.append([distance[idx, i], int(i), y[i]])
        distance_sort.sort(key=lambda x: x[0])

        for i in range(n_samples):
            # find k nearest hit points
            if distance_sort[i][2] == y[idx]:
                if len(near_hit) < k:
                    near_hit.append(distance_sort[i][1])
                elif len(near_hit) == k:
                    stop_dict[y[idx]] = 1
            else:
                # find k nearest miss points for each label
                if len(near_miss[distance_sort[i][2]]) < k:
                    near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                else:
                    if len(near_miss[distance_sort[i][2]]) == k:
                        stop_dict[distance_sort[i][2]] = 1
            stop = True
            for (key, value) in stop_dict.items():
                    if value != 1:
                        stop = False
            if stop:
                break

        # update reliefF score
        near_hit_term = np.zeros(n_features)
        for ele in near_hit:
            near_hit_term = np.array(abs(self_fea-X[ele, :]))+np.array(near_hit_term)

        near_miss_term = dict()
        for (label, miss_list) in near_miss.items():
            near_miss_term[label] = np.zeros(n_features)
            for ele in miss_list:
                near_miss_term[label] = np.array(abs(self_fea-X[ele, :]))+np.array(near_miss_term[label])
            score += near_miss_term[label]/(k*p_dict[label])
        score -= near_hit_term/k
    return score


def feature_ranking(score):
    """
    Rank features in descending order according to reliefF score, the higher the reliefF score, the more important the
    feature is
    """
    idx = np.argsort(score, 0)
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

        feature_score = reliefF(X_train,y_train)
        wd.out_feature_score('reliefF', feature_score, classifier, file_name, count, size)
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
    print(selected_num,'\t', 'reliefF', '\t', acc)
    return acc




