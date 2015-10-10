from numpy import *
def preProcess(X):
    stdX = std(X, dtype = float, ddof = 1, axis = 0)
    idx1 = (stdX != 0)
    centrX = X - mean(X, axis = 0)
    tmp = centrX[:, idx1] / stdX[idx1]
    X[:, idx1] = tmp
    X[:, idx1] = (X[:, idx1] - mean(X[:, idx1] ,axis = 0))/std(X[:, idx1], dtype = float, ddof = 1, axis = 0)
    X = X/sqrt(sum(X*X, axis = 1)).reshape(X.shape[0], 1)
    return X


# X = array([[ 1,  6,  4, 12,  5,  5,  3,  4,  1, 67,  3,  2,  1,  2,  1,  0,  0,
#          1,  0,  0,  1,  0,  0,  1],
#        [ 2, 48,  2, 60,  1,  3,  2,  2,  1, 22,  3,  1,  1,  1,  1,  0,  0,
#          1,  0,  0,  1,  0,  0,  1],
#        [ 4, 12,  4, 21,  1,  4,  3,  3,  1, 49,  3,  1,  2,  1,  1,  0,  0,
#          1,  0,  0,  1,  0,  1,  0]], dtype = float)

# X = preProcess(X)
# print X
