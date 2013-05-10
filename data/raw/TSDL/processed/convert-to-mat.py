import scipy.io
import os
import numpy as np

for filename in os.listdir('.'):
    if os.path.splitext(filename)[-1] == '.csv':
        data = np.genfromtxt(filename, delimiter=',')
        if data.shape[0] > 1000:
            print 'Subsampling %s' % filename
            subset = np.random.choice(range(data.shape[0]), size=1000, replace=False)
            data = data[sorted(subset),:]
        scipy.io.savemat(os.path.splitext(filename)[0] + '.mat', {'X' : data[:,0], 'y' : data[:,1]})
