import skfeature.utility.entropy_estimators as ee
import numpy as np
import LoadingData as ld

def information_gain(f1, f2):
    """
    This function calculates the information gain, where ig(f1,f2) = H(f1) - H(f1|f2)

    Input
    -----
    f1: {numpy array}, shape (n_samples,)
    f2: {numpy array}, shape (n_samples,)

    Output
    ------
    ig: {float}
    """

    ig = ee.entropyd(f1) - conditional_entropy(f1, f2)
    return ig


def conditional_entropy(f1, f2):
    """
    This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)

    Input
    -----
    f1: {numpy array}, shape (n_samples,)
    f2: {numpy array}, shape (n_samples,)

    Output
    ------
    ce: {float}
        ce is conditional entropy of f1 and f2
    """

    ce = ee.entropyd(f1) - ee.midd(f1, f2)
    return ce


def su_calculation(f1, f2):
    """
    This function calculates the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))

    Input
    -----
    f1: {numpy array}, shape (n_samples,)
    f2: {numpy array}, shape (n_samples,)

    Output
    ------
    su: {float}
        su is the symmetrical uncertainty of f1 and f2

    """

    # calculate information gain of f1 and f2, t1 = ig(f1,f2)
    t1 = information_gain(f1, f2)
    # calculate entropy of f1, t2 = H(f1)
    t2 = ee.entropyd(f1)
    # calculate entropy of f2, t3 = H(f2)
    t3 = ee.entropyd(f2)
    # su(f1,f2) = 2*t1/(t2+t3)
    su = 2.0*t1/(t2+t3)

    return su

if __name__ == '__main__':
    data_other = ld.openfile('E:/1_Code/FatureSelection/1_semi_supervised_feature_selection/dataset/data_small/glass.csv')
    X_other, y_other = ld.splitData(data_other)
    m = np.array([[8,62,3],[4,12,0],[3,5,9],[1,8,0],[78,4,26],[76,2,0],[86,23,56],[0,5,9],[0,5,0],[0,5,0],[0,5,0]])
    _,ncol = X_other.shape
    print(X_other)
    igM = np.zeros((ncol,ncol))
    sumIG = []
    for i in range(ncol - 1):
        for j in range(i + 1,ncol):
            igM[i,j] = information_gain(X_other[:,i],X_other[:,j])
            igM[j,i] = igM[i,j]
    for i in range(ncol):
        sumIG.append(igM[i,:].sum())
    print(igM)
    print(sumIG)