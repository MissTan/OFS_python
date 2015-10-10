from numpy import *
from pandas import read_csv
from preProcess import preProcess
from OFSGD import OFSGD
import matplotlib.pyplot as plt
class options:
    NumFeature = 0

####laad data
X = read_csv('./german_data.csv', header = None, dtype = float)
X = array(X)
Y = read_csv('./german_label.csv', header = None, dtype = float)
Y = array(Y)
id_list = read_csv('./ID.csv', header = None, dtype = float)
id_list = array(id_list)
print 'load data finished'


###standerized the data
X = preProcess(X)

print 'preProcess finished'

###run OFSGD
(n, d) = X.shape
mistakes_list = []
options.NumFeature=max(1,round(0.1*(d-1)));
for i in range(len(id_list)):
    mistakes = OFSGD(X, Y, options, id_list[i])
    mistakes_list.append(mistakes)


mistakes = mean(mistakes_list, axis = 0)

###plot
plt.figure(figsize=(8,4))
plt.plot(mistakes[10:],label="OFS",color="red",linewidth=2)
plt.xlabel("iteration")
plt.ylabel("mis rates")
plt.legend()
plt.show()
plt.save('mistakes')
