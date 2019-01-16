import numpy as np
import math as math
import LoadingData as ld


# 求任意两个样本的相似度
def getSimilarity(a, b,type = 'Euclidean'):
    # calculate similarity between a and b (a and b shall be 1*d matrix)
    w = 0.0
    if type == 'Euclidean':
        a = np.array(a)
        b = np.array(b)
        w = math.sqrt(((a - b) ** 2).sum())  # 欧氏距离
    if type == 'Manhattan':
        a = np.array(a)
        b = np.array(b)
        w = np.abs(a - b).sum()
    if type == 'Chebyshev':
        a = np.array(a)
        b = np.array(b)
        w = np.abs(a - b).max()

    return w


# 求邻接矩阵，是一个方阵，大小为：样本数*样本数
def getAdjacencyMatrix(data_matrix,type = 'Euclidean'):
    data_matrix = np.mat(data_matrix)
    nrow = data_matrix.shape[0]
    adjacency_matrix = np.zeros((nrow, nrow))
    for row in range(nrow - 1):
        for col in range(row + 1, nrow):
            adjacency_matrix[row][col] = getSimilarity(data_matrix[row], data_matrix[col],type)
            adjacency_matrix[col][row] = adjacency_matrix[row][col]

    return adjacency_matrix


# 求度矩阵
def getDegreeMatrix(adjacency_matrix):
    adjacency_matrix = np.mat(adjacency_matrix)
    degree_matrix = adjacency_matrix.sum(axis=1).A1
    degree_matrix = np.mat(np.diag(degree_matrix))
    for d in range(degree_matrix.shape[0]):
        if degree_matrix[d, d] == 0:
            degree_matrix[d, d] = 0.001

    return degree_matrix


# 求拉普拉斯矩阵
def getLaplacianMatrix(adjacency_matrix, normalize=True):
    adjacency_matrix = np.mat(adjacency_matrix)
    nrow = adjacency_matrix.shape[0]
    ncol = adjacency_matrix.shape[1]
    degree_matrix = getDegreeMatrix(adjacency_matrix)
    # L = D - W
    laplacian_matrix = degree_matrix - adjacency_matrix
    # L = D^(-1/2) L D^(-1/2)
    if normalize:
        sqrt_deg_matrix = np.mat(np.diag(np.diag(degree_matrix) ** (-0.5)))
        laplacian_matrix = sqrt_deg_matrix * laplacian_matrix * sqrt_deg_matrix

    return laplacian_matrix


# 求拉普拉斯矩阵谱
def getSpectrum(adjacency_matrix, normalize=True):
    # 返回特征值和对应的特征向量(each column an eigen vector)(matrix)
    adjacency_matrix = np.mat(adjacency_matrix)
    laplacian_matrix = getLaplacianMatrix(adjacency_matrix, normalize)
    (eigen_value, eigne_vector) = np.linalg.eig(laplacian_matrix)
    index = eigen_value.argsort()
    eigen_value.sort()
    eigen_vector = eigne_vector[:, index]

    return eigen_value, eigen_vector


if __name__ == '__main__':
    file_name = 'dataset\iris.csv'
    title_names, data_matrix = ld.import_matrix(file_name, has_title_line=True)
    adjacency_matrix = getAdjacencyMatrix(data_matrix)
    degree_matrix = getDegreeMatrix(adjacency_matrix)
    sqrt_deg_matrix = np.mat(np.diag(np.diag(degree_matrix) ** (-0.5)))
    laplacian_matrix = getLaplacianMatrix(adjacency_matrix, normalize=True)
    (eigen_value, eigen_vector) = getSpectrum(adjacency_matrix, normalize=True)
    print(str(eigen_value.shape[0]) + ' eigen_value:')
    print(eigen_value)
    print('eigen_vector(' + str(eigen_vector.shape[0]) + '*' + str(eigen_vector.shape[0]) + '):')
    print(eigen_vector)
