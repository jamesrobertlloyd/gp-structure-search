import scipy.io
import os
import numpy as np

for filename in os.listdir('.'):
    if os.path.splitext(filename)[-1] == '.csv':
        data = np.genfromtxt(filename, delimiter=',')
        scipy.io.savemat(os.path.splitext(filename)[0] + '.mat', {'X' : data[:,0], 'y' : data[:,1]})
