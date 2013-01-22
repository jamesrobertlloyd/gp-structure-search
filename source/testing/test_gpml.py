'''
Some routines to test GPML.

@authors: 
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          David Duvenaud (dkd23@cam.ac.uk)
'''

import gpml
import flexiblekernel as fk

import os
import tempfile
import scipy.io

#### FIXME - change these into tests again

def load_mauna():
    data_file = '../data/mauna.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']

def not_tet_num_params():
    '''The number of hyper-parameters passed in to gpml should be the same as the number returned.'''
    k = fk.SqExpKernel(0, 0)
    init_params = k.param_vector()
    X, y = load_mauna()
    optimized_hypers, _ = gpml.optimize_params(k.gpml_kernel_expression(), init_params, X, y)
    assert optimized_hypers.size == init_params.size
    
def not_tet_matlab_stops():
    '''run_matlab_code should stop and report an error if its code causes Matlab to throw an exception.'''
    try:
        gpml.run_matlab_code("asdf")
    except Exception:
        pass
    else:
        raise AssertionError('Failure expected')
    
def not_tet_matlab_runs():
    '''run_matlab_code should execute the Matlab code.'''
    fname = tempfile.mkstemp(suffix='.mat')[1]
    code = "a = 2 + 2; save('%s');" % fname
    gpml.run_matlab_code(code)
    v = scipy.io.loadmat(fname)
    os.remove(fname)
    assert v['a'] == 4
    
    
    
