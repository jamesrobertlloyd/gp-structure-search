import numpy as np
nax = np.newaxis
import scipy.io
import subprocess, os

import config

# Matlab code to optimize hyperparams on one file, given one kernel.
OPTIMIZE_KERNEL_CODE = r"""
% Load the data, it should contain X and y.
load %(datafile)s

% Load GPML
addpath(genpath(%(gpml_path)s));

% Set up model.
meanfunc = {@meanConst}
hyp.mean = mean(y)

covfunc = %(kernel_family)s
hyp.cov = %(kernel_params)s

likfunc = @likGauss
hyp.lik = log(std(y)/10)

% Optimize hyperparameters.
[hyp_opt, nlls] = minimize(hyp2, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
best_nll = nlls(end)

% Save results to a file.
save( '%(writefile)s', 'hyp_opt', 'best_nll' );
"""

data_file = 'mauna.txt'

#code = OPTIMIZE_KERNEL_CODE % {'dir': os.curdir(),
#                               'datafile': data_file}


def run_matlab_code(code):
    call = [config.MATLAB_LOCATION, '-nosplash', '-nojvm', '-nodisplay', '-r', code]
    subprocess.call(call)

def optimize_params(kernel_family, kernel_init_params, X, y):
    if X.ndim == 1:
        X = X[:, nax]
    if y.ndim == 1:
        y = y[:, nax]
    data = {'X': X, 'y': y}
    scipy.io.savemat(config.TEMP_DATA_FILE, data)

    code = OPTIMIZE_KERNEL_CODE % {'datafile': config.TEMP_DATA_FILE,
                                   'writefile': config.TEMP_WRITE_FILE,
                                   'kernel_family': kernel_family,
                                   'kernel_params': kernel_init_params}
    run_matlab_code(code)


