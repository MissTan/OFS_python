from pandas import read_csv
from scipy.sparse import bsr_matrix
from numpy import array

def loadSparseData(fname, nr):
    data = read_csv('./COML_data/' + fname + '_data.csv', header = None, dtype = float)
    label = read_csv('./COML_data/' + fname + '_label.csv', header = None, dtype = float)
    X = bsr_matrix((data[2], (data[0], data[1]))).toarray()
    X = array(X)[:nr, :]
    Y = array(label)[:nr, :]
    print 'load ', fname, ': ', X.shape

    return X, Y

