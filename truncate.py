from numpy import *

def truncate(w, NumFeature):
    w = transpose(w)
    if sum(w != 0) > NumFeature:
        ids = argsort(abs(w))
        ids = ids[:, :len(w) - NumFeature - 1 ]
        w[:, ids] = 0
    w = transpose(w)
    return w




# ww = array([[ 0.  ],
#        [ 0.9 ],
#        [-2.16],
#        [ 0.52],
#        [-2.24],
#        [-1.36],
#        [-0.62],
#        [ 0.12],
#        [ 0.72],
#        [-0.52],
#        [-0.98],
#        [-0.1 ],
#        [ 0.1 ],
#        [-0.16],
#        [-0.14],
#        [ 0.44],
#        [-0.3 ],
#        [-0.02],
#        [ 0.02],
#        [ 0.22],
#        [-0.36],
#        [ 0.02],
#        [ 0.14],
#        [-0.38]])

# truncate(ww, 6)
# print ww
