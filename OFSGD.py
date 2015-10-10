from truncate import truncate
from numpy import *

def OFSGD(X, Y, options, id_list):
    NumFeature = options.NumFeature
    ID = id_list
    error_count = 0
    mistakes = []
    error_list  = []
    eta = 0.2
    lbda = 0.01
    k = 2

    w = zeros([X.shape[1], 1])

    for t in xrange(len(ID)):

        id = ID[t]
        x_t = transpose(X[id - 1, :])
        x_t = x_t.reshape(len(x_t), 1)
        f_t = dot(transpose(w), x_t)
        y_t = Y[id - 1]

        if y_t * f_t < 0:
            error_count += 1
            error_list.append(id)

        if y_t * f_t <= 1:
            delt = (eta * y_t * x_t)
            w = w + delt
            w = dot(w, min(1, 1 / (sqrt(lbda) * linalg.norm(w))))
            truncate(w, NumFeature)

        if t % 10 == 1:
            mistakes.append(error_count/float(t))

    return mistakes, error_list


