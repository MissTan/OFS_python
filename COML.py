from numpy import *
from pandas import read_csv
import matplotlib.pyplot as plt

# X1 = array([ [1, 6, 4, 12, 5, 5, 3, 4, 1, 67],
#             [4, 12, 4, 21, 1, 4, 3, 3, 1, 49],
#             [1, 42, 2, 79, 1, 4, 3, 4, 2, 45],
#             [4, 36, 2, 91, 5, 3, 3, 4, 4, 35],
#             [4, 24, 2, 28, 3, 5, 3, 4, 2, 53]])

# X2 = array([[2, 15, 1, 13, 2, 3, 4, 2, 2, 25],
#             [2, 30, 4, 84, 1, 4, 3, 2, 2, 49],
#             [4, 48, 2, 48, 1, 1, 3, 2, 3, 33],
#             [1, 36, 2, 82, 1, 3, 3, 2, 2, 26],
#             [1, 45, 2, 18, 1, 3, 3, 4, 4, 23]])
fname = 'german'

X1 = read_csv('./COML_data/' + fname + '_pos.csv', header = None, dtype = float)
X1 = array(X1)
X2 = read_csv('./COML_data/' + fname + '_neg.csv', header = None, dtype = float)
X2 = array(X2)

Y1 = 1
Y2 = -1

#def COML(X, Y):
if 1:
    C = 1
    T = min(len(X1), len(X2))
    w1 = zeros([X1.shape[1], 1])
    w2 = zeros([X2.shape[1], 1])
    u = zeros([X2.shape[1], 1])
    fai1 = 0.5
    fai2 = 0.5
    error_count = 0
    mistake_list = []
    for i in range(T):
        #print 'i = ', i
        # update collaborative models
        x1_t = X1[i]
        #print 'x1_t', x1_t
        f_t1 = dot(x1_t, w1)
        #print 'f_t1', f_t1

        loss_w1 = max(0, 1 - Y1 * f_t1)
        if loss_w1 == 0:
            tau = 0
        else:
            tau = min(C, (fai1+fai2-dot(x1_t, Y1*(fai1*w1+fai2*u))/linalg.norm(x1_t)))
        #print 'tau = ', tau
        w1 = fai1 * w1 + fai2 * u + tau * Y1 * x1_t.reshape(len(x1_t), 1)
        #print 'w1', transpose(w1)
        #update w2
        x2_t = X2[i]
        #print 'x2_t', x2_t
        f_t2 = dot(x2_t, w2)
        #print 'f_t2', f_t2

        loss_w2 = max(0, 1 - Y2 * f_t2)
        if loss_w2 == 0:
            tau = 0
        else:
            tau = min(C, (fai1+fai2-dot(x2_t, Y2*(fai1*w2+fai2*u))/linalg.norm(x2_t)))
        #print 'tau', tau
        w2 = fai1 * w2 + fai2 * u + tau * Y2 * x2_t.reshape(len(x2_t), 1)
        #print 'w2', transpose(w2)
        #update global models
        x1_t = X1[i]
        fu_t1 = dot(x1_t, u)
        loss_u = max(0, 1 - Y1 * fu_t1)
        if loss_u > 0:
            tau = min(C, (1 - Y1*fu_t1)/linalg.norm(x1_t))
            u = u + tau * Y1 * x1_t.reshape(len(x1_t), 1)

        x2_t = X2[i]
        fu_t2 = dot(x2_t, u)
        loss_u = max(0, 1 - Y2 * fu_t2)
        if loss_u > 0:
            tau = min(C, (1 - Y2*fu_t2)/linalg.norm(x2_t))
            u = u + tau * Y2 * x2_t.reshape(len(x2_t), 1)

        if Y1 * f_t1 <= 0:
            error_count += 1
        if Y2 * f_t2 <= 0:
            error_count += 1

        if i%10 == 1:
            mistake_list.append(error_count/float(2*i))

print mistake_list
plt.figure(figsize=(8,4))
plt.plot(mistake_list, label="OFS", color="red",linewidth=2)
plt.show()
