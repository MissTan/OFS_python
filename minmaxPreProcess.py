from numpy import *

def mmPreProcess(X):
    centrX = X - X.min(axis = 0)
    minmaxX = X.max(axis = 0) - X.min(axis = 0)
    minmaxX += 0.00001
    X = centrX / minmaxX

    return X



# X = array([[1, 2, 3, 4], [2, 2, 4, 5], [6, 2, 8, 9]], dtype = float)
# print X
# X = mmPreProcess(X)


