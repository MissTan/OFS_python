from numpy import *
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.sparse import bsr_matrix
from stdPreProcess import stdPreProcess
from loadSparseData import loadSparseData
from PA import PAlist
from minmaxPreProcess import mmPreProcess
import random

def update_CM(i, X, Y, w, u):
    x_t = X[i]
    f_t = dot(w, x_t)
    loss_w = max(0, 1 - Y[i] * f_t)
    if loss_w == 0:
        tau = 0
    else:
        tau = min(C, (fai1+fai2-Y[i]*dot((fai1*w+fai2*u), x_t)/(linalg.norm(x_t)**2)))
    w = fai1 * w + fai2 * u + tau * Y[i] * x_t.reshape(1, len(x_t))
    return w, f_t

def update_GM(i, X, Y, w, u):
    x_t = X[i]
    fu_t = dot(u, x_t)
    # print 'Y[i]', Y[i]
    # print 'fu_t', fu_t
    loss_u = max(0, 1 - Y[i] * fu_t)
    if loss_u > 0:
        tau = min(C, (1 - Y[i]*fu_t)/(linalg.norm(x_t)**2))
        u = u + tau * Y[i] * x_t.reshape(1, len(x_t))
    return u, fu_t

def COML(X_list, Y_list, fai1, fai2, alf, beta, ID_list):
    C = 1
    T = max([len(x) for x in X_list])
    w_list = [zeros([1, X.shape[1]]) for X in X_list]
    u = zeros([1, X2.shape[1]])
    ft_list = [zeros([1, X.shape[1]]) for X in X_list]
    fut_list = [zeros([1, X.shape[1]]) for X in X_list]
    len_list = [len(x) for x in X_list]
    errcount_list = zeros([len(X_list), 1])
    misrate_list = [[],[],[],[]]
    count_list = zeros([len(X_list), 1])
    for i in range(T):
        #update user model
        for j in range(len(X_list)):
            if i < len_list[j]:
                idx = ID_list[j][i]
                w_list[j], ft_list[j] = update_CM(idx, X_list[j], Y_list[j], w_list[j], u)
                count_list[j] += 1

        #update golbal model
        for j in range(len(X_list)):
            if i < len_list[j]:
                idx = ID_list[j][i]
                u, fut_list[j] = update_GM(idx, X_list[j], Y_list[j], w_list[j], u)

        for j in range(len(X_list)):
            if i < len_list[j]:
                idx = ID_list[j][i]
                if  Y_list[j][idx] * (alf * ft_list[j] + beta * fut_list[j]) < 0:
                    errcount_list[j] +=1
                misrate_list[j].append(errcount_list[j]/float(count_list[j]))

    return misrate_list, w_list, u


# load data
fname1 = 'tfidf_pu1'
fname2 = 'tfidf_pu2'
fname3 = 'tfidf_pu3'
fname4 = 'tfidf_pua'
sampleNum = -1
X1, Y1 = loadSparseData(fname1, sampleNum)
X2, Y2 = loadSparseData(fname2, sampleNum)
X3, Y3 = loadSparseData(fname3, sampleNum)
X4, Y4 = loadSparseData(fname4, sampleNum)

# preProcess data
print 'preProcess X'
X1 = mmPreProcess(X1)
X2 = mmPreProcess(X2)
X3 = mmPreProcess(X3)
X4 = mmPreProcess(X4)

# X1 = stdPreProcess(X1)
# X2 = stdPreProcess(X2)
# X3 = stdPreProcess(X3)
# X4 = stdPreProcess(X4)

# partition data to train and test
trainRatio = 0.9
bar1 = int(X1.shape[0] * trainRatio)
bar2 = int(X2.shape[0] * trainRatio)
bar3 = int(X3.shape[0] * trainRatio)
bar4 = int(X4.shape[0] * trainRatio)

X_list = []
X_list.append(X1[:bar1][:])
X_list.append(X2[:bar2][:])
X_list.append(X3[:bar3][:])
X_list.append(X4[:bar4][:])

Y_list = []
Y_list.append(Y1)
Y_list.append(Y2)
Y_list.append(Y3)
Y_list.append(Y4)

ID_list = []
idl1 = range(bar1)
idl2 = range(bar2)
idl3 = range(bar3)
idl4 = range(bar4)
random.shuffle(idl1)
random.shuffle(idl2)
random.shuffle(idl3)
random.shuffle(idl4)
ID_list.append(idl1)
ID_list.append(idl2)
ID_list.append(idl3)
ID_list.append(idl4)

# COML train
print 'start COML'
fai1 = 0.5
fai2 = 0.5
alf = 0.5
beta = 0.5
C = 1
COML_mis, COML_w, u = COML(X_list, Y_list, fai1, fai2, alf, beta, ID_list)
PA_mis, PA_w = PAlist(X_list, Y_list, ID_list)

# show
plt.figure(figsize=(16,8))
for j in range(len(X_list)):
    plt.subplot(len(X_list), 1, j)
    plt.plot(COML_mis[j][50::10], color="blue")
    plt.plot(PA_mis[j][50::10], color="red")
plt.show()
