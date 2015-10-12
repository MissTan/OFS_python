from numpy import *
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.sparse import bsr_matrix
from preProcess import preProcess
from loadSparseData import loadSparseData

fname1 = 'pu1'
fname2 = 'pu2'
fname3 = 'pu3'
fname4 = 'pua'
X1, Y1 = loadSparseData(fname1)
X2, Y2 = loadSparseData(fname2)
X3, Y3 = loadSparseData(fname3)
X4, Y4 = loadSparseData(fname4)

print 'preProcess X'
X1 = preProcess(X1)
X2 = preProcess(X2)
X3 = preProcess(X3)
X4 = preProcess(X4)

print 'start COML'

def update_CM(i, X, Y, w, u):
    x_t = X[i]
    f_t = dot(w, x_t)
    loss_w = max(0, 1 - Y[i] * f_t)
    if loss_w == 0:
        tau = 0
    else:
        tau = min(C, (fai1+fai2-dot(Y[i]*(fai1*w+fai2*u), x_t)/linalg.norm(x_t)))
    w = fai1 * w + fai2 * u + tau * Y[i] * x_t.reshape(1, len(x_t))
    return w, f_t

def update_GM(i, X, Y, w, u):
    x_t = X[i]
    fu_t = dot(u, x_t)
    loss_u = max(0, 1 - Y[i] * fu_t)
    if loss_u > 0:
        tau = min(C, (1 - Y[i]*fu_t)/linalg.norm(x_t))
        u = u + tau * Y[i] * x_t.reshape(1, len(x_t))
    return u, fu_t

if 1:
    C = 1
    T = max(len(X1), len(X2), len(X3), len(X4))
    w1 = zeros([1, X1.shape[1]])
    w2 = zeros([1, X2.shape[1]])
    w3 = zeros([1, X3.shape[1]])
    w4 = zeros([1, X4.shape[1]])
    u = zeros([1, X2.shape[1]])
    len1 = X1.shape[0]
    len2 = X2.shape[0]
    len3 = X3.shape[0]
    len4 = X4.shape[0]

    fai1 = 0
    fai2 = 1
    error_count = 0
    error1 = 0
    error2 = 0
    error3 = 0
    error4 = 0
    er1_list = []
    er2_list = []
    er3_list = []
    er4_list = []
    mistake_list = []
    cal_count = 0
    cal1 = 0
    cal2 = 0
    cal3 = 0
    cal4 = 0
    for i in range(T):
        ## colabrative model
        if i < len1:
            w1, f_t1 = update_CM(i, X1, Y1, w1, u)
            cal1 += 1
            cal_count += 1

        if i < len2:
            w2, f_t2 = update_CM(i, X2, Y2, w2, u)
            cal2 += 1
            cal_count += 1


        if i < len3:
            w3, f_t3 = update_CM(i, X3, Y3, w3, u)
            cal3 += 1
            cal_count += 1

        if i < len4:
            w4, f_t4 = update_CM(i, X4, Y4, w4, u)
            cal4 += 1
            cal_count += 1

        # global model
        if i < len1:
            u, fu_t1 = update_GM(i, X1, Y1, w1, u)

        if i < len2:
            u, fu_t2 = update_GM(i, X2, Y2, w2, u)

        if i < len3:
            u, fu_t3 = update_GM(i, X3, Y3, w3, u)

        if i < len4:
            u, fu_t4 = update_GM(i, X4, Y4, w4, u)


        # caculate error count
        alf = 1
        beta = 0
        if i < len1:
            if Y1[i] * (alf * f_t1 + beta * fu_t1) <= 0:
                error1 += 1
                error_count += 1
        if i < len2:
            if Y2[i] * (alf * f_t2 + beta * fu_t2) <= 0:
                error2 += 1
                error_count += 1
        if i < len3:
            if Y3[i] * (alf * f_t3 + beta * fu_t3) <= 0:
                error3 += 1
                error_count += 1
        if i < len4:
            if Y4[i] * (alf * f_t4 + beta * fu_t4) <= 0:
                error4 += 1
                error_count += 1


        if i%10 == 1:
            mistake_list.append(error_count/float(cal_count))
            er1_list.append(error1/float(cal1)
                )
            er2_list.append(error2/float(cal2)
                )
            er3_list.append(error3/float(cal3)
                )
            er4_list.append(error4/float(cal4)
                )
print mistake_list
plt.figure(figsize=(8,4))
plt.plot(mistake_list, label="OFS", color="red",linewidth=2)
plt.plot(er1_list, label="OFS", color="green",linewidth=2)
plt.plot(er2_list, label="OFS", color="blue",linewidth=2)
plt.plot(er3_list, label="OFS", color="yellow",linewidth=2)
plt.plot(er4_list, label="OFS", color="black",linewidth=2)
plt.show()

