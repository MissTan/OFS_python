from truncate import truncate
from numpy import *
from loadSparseData import loadSparseData
from stdPreProcess import stdPreProcess
import matplotlib.pyplot as plt
from loadData import loadData


eta = 0.2
lbda = 0.01
k = 2

def OFSGD_update(x_t, y_t, NumFeature, w):
    f_t = dot(w, x_t)
    if y_t * f_t <= 1:
        delt = (eta * y_t * x_t)
        w = w + delt
        #w = dot(w, min(1, 1 / (sqrt(lbda) * linalg.norm(w))))
        w = w * min(1, 1 / (sqrt(lbda) * linalg.norm(w)))
        truncate(w, NumFeature)
    else:
        w = w
    return w, f_t

def OFSGD(X, Y, id_list, rateFeature):
    NumFeature = max(1, round(rateFeature * X.shape[1]))
    w = zeros([1, X.shape[1]])
    error = 0
    mistakes = []
    error_list = []
    f_list = []
    for t in range(X.shape[0]):
        if t % 100 == 0:
            print 'in rount', t
        id = id_list[t]
        x_t = X[id, :]
        y_t = Y[id]
        w, f_t = OFSGD_update(x_t, y_t, NumFeature, w)
        if y_t * f_t <= 0:
            error += 1
            error_list.append(id)
            f_list.append(f_t)
        mistakes.append(error/float(t + 1))
    return mistakes, w

# sparse data demo
# fname1 = 'pu1'
# sampleNum = 700
# X, Y = loadSparseData(fname1, sampleNum)

# print 'process data'
# X = stdPreProcess(X)

# id_list = range(len(X))

# rateFeature = 0.1
# mistakes, w = OFSGD(X, Y, id_list, rateFeature)


# plt.figure(figsize=(16,8))
# plt.subplot(4, 1, 1)
# plt.plot(mistakes[1:], color="blue")
# plt.show()

# dense data
# fname1 = 'magic04'
# X, Y, ID = loadData(fname1)

# print 'process data'
# X = stdPreProcess(X)
# id_list = ID[0] - 1
# rateFeature = 0.1
# mistakes, w = OFSGD(X, Y, id_list, rateFeature)


# #######################

# plt.figure(figsize=(16,8))
# plt.subplot(1, 1, 1)
# plt.plot(mistakes[1:], color="blue")
# plt.show()
