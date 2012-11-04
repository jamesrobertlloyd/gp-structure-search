import numpy as np
nax = np.newaxis
import scipy.io
from subprocess import Popen, PIPE, STDOUT

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

[hyp_opt, nlls] = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, X, y);
best_nll = nlls(end)

save( '%(writefile)s', 'hyp_opt', 'best_nll' );
exit();
"""


def run_matlab_code(code):
    call = [config.MATLAB_LOCATION, '-nosplash', '-nojvm', '-nodisplay']
    p = Popen(call, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    grep_stdout = p.communicate(input=code)[0]
    print(grep_stdout)
    #subprocess.call(call)

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


