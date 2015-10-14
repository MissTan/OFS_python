from truncate import truncate
from numpy import *
from loadSparseData import loadSparseData
import matplotlib.pyplot as plt
from minmaxPreProcess import mmPreProcess


def PA(X, Y, ID):
    error_count = 0
    mistakes = []
    error_list  = []
    eta = 0.2
    lbda = 0.01
    k = 2

    w = zeros([X.shape[1], 1])

    for t in xrange(len(ID)):
        id = ID[t]
        x_t = X[id]
        f_t = dot(w.reshape(1, len(w)), x_t)
        y_t = Y[id]
        loss = max(0, 1 - y_t * f_t)

        if y_t * f_t <= 0:
            error_count += 1
            error_list.append(id)

        if y_t * f_t <= 1:
            tau = loss/(linalg.norm(x_t)**2)
            w = w + tau * y_t * x_t.reshape(len(w), 1)

        mistakes.append(error_count/float(t+1))

    return mistakes, w

def PAlist(X_list, Y_list, ID_list):
    mis_list = []
    w_list = []
    for i in range(len(X_list)):
        mistakes, w = PA(X_list[i], Y_list[i], ID_list[i])
        mis_list.append(mistakes)
        w_list.append(w)
    return mis_list, w_list


# PA
# fname1 = 'pu1'
# X1, Y1 = loadSparseData(fname1)

# ID1 = range(len(X1))
# mistakes1 = PA(X1, Y1, ID1)

# print mistakes1
# plt.figure(figsize=(8,4))
# plt.plot(mistakes1, label="OFS", color="red",linewidth=2)
# plt.show()

# PAlist
# fname1 = 'tfidf_pu1'
# fname2 = 'tfidf_pu2'
# fname3 = 'tfidf_pu3'
# fname4 = 'tfidf_pua'
# sampleNum = -1
# X1, Y1 = loadSparseData(fname1, sampleNum)
# X2, Y2 = loadSparseData(fname2, sampleNum)
# X3, Y3 = loadSparseData(fname3, sampleNum)
# X4, Y4 = loadSparseData(fname4, sampleNum)

# # preProcess data
# print 'preProcess X'
# X1 = mmPreProcess(X1)
# X2 = mmPreProcess(X2)
# X3 = mmPreProcess(X3)
# X4 = mmPreProcess(X4)

# # X1 = stdPreProcess(X1)
# # X2 = stdPreProcess(X2)
# # X3 = stdPreProcess(X3)
# # X4 = stdPreProcess(X4)

# # partition data to train and test
# trainRatio = 0.9
# bar1 = int(X1.shape[0] * trainRatio)
# bar2 = int(X2.shape[0] * trainRatio)
# bar3 = int(X3.shape[0] * trainRatio)
# bar4 = int(X4.shape[0] * trainRatio)

# X_list = []
# X_list.append(X1[:bar1][:])
# X_list.append(X2[:bar2][:])
# X_list.append(X3[:bar3][:])
# X_list.append(X4[:bar4][:])

# Y_list = []
# Y_list.append(Y1)
# Y_list.append(Y2)
# Y_list.append(Y3)
# Y_list.append(Y4)

# ID_list = []
# idl1 = range(bar1)
# idl2 = range(bar2)
# idl3 = range(bar3)
# idl4 = range(bar4)
# random.shuffle(idl1)
# random.shuffle(idl2)
# random.shuffle(idl3)
# random.shuffle(idl4)
# ID_list.append(idl1)
# ID_list.append(idl2)
# ID_list.append(idl3)
# ID_list.append(idl4)

# mis_list, w = PAlist(X_list, Y_list, ID_list)

# plt.figure(figsize=(16,8))
# for j in range(len(X_list)):
#     plt.subplot(len(X_list), 1, j)
#     plt.plot(mis_list[j][50::10], color="blue")
# plt.show()
