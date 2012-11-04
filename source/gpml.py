import numpy as np
nax = np.newaxis
import scipy.io
import tempfile, os
import subprocess

import config

# Matlab code to optimize hyperparams on one file, given one kernel.
OPTIMIZE_KERNEL_CODE = r"""
a='Load the data, it should contain X and y.'
load '%(datafile)s'

a='Load GPML'
addpath(genpath('%(gpml_path)s'));

a='Set up model.'
meanfunc = {@meanConst}
hyp.mean = mean(y)

covfunc = %(kernel_family)s
hyp.cov = %(kernel_params)s

likfunc = @likGauss
hyp.lik = log(std(y)/10)

[hyp_opt, nlls] = minimize(hyp, @gp, -300, @infExact, meanfunc, covfunc, likfunc, X, y);
best_nll = nlls(end)

save( '%(writefile)s', 'hyp_opt', 'best_nll' );
exit();
"""


def run_matlab_code(code):
    # Write to a temp script
    script_file = tempfile.mkstemp(suffix='.m')[1]
    stdout_file = tempfile.mkstemp(suffix='.txt')[1]
    stderr_file = tempfile.mkstemp(suffix='.txt')[1]
    
    open(script_file, 'w').write(code)
    
    call = [config.MATLAB_LOCATION, '-nosplash', '-nojvm', '-nodisplay']
    subprocess.call(call, stdin=open(script_file), stdout=open(stdout_file, 'w'), stderr=open(stderr_file, 'w'))
    
    err_txt = open(stderr_file).read()
    
    os.remove(script_file)
    os.remove(stdout_file)
    os.remove(stderr_file)
    
    if err_txt != '':
        raise RuntimeError('Matlab produced the following errors:\n\n%s' % err_txt)
    
    

def optimize_params(kernel_expression, kernel_init_params, X, y):
    if X.ndim == 1:
        X = X[:, nax]
    if y.ndim == 1:
        y = y[:, nax]
    data = {'X': X, 'y': y}
    scipy.io.savemat(config.TEMP_DATA_FILE, data)

    code = OPTIMIZE_KERNEL_CODE % {'datafile': config.TEMP_DATA_FILE,
                                   'writefile': config.TEMP_WRITE_FILE,
                                   'gpml_path': config.GPML_PATH,
                                   'kernel_family': kernel_expression,
                                   'kernel_params': kernel_init_params}
    run_matlab_code(code)

    # Load in the file that GPML saved things to.
    gpml_result = scipy.io.loadmat(config.TEMP_WRITE_FILE)

    optimized_hypers = gpml_result['hyp_opt']
    nll = gpml_result['best_nll'][0, 0]

    # Strip out only kernel hyperparameters.
    kernel_hypers = optimized_hypers['cov'][0, 0].ravel()

    return kernel_hypers, nll


