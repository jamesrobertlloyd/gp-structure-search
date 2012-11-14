'''
Some routines to interface with GPML.

@authors: 
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          David Duvenaud (dkd23@cam.ac.uk)
'''

import numpy as np
nax = np.newaxis
import scipy.io
import tempfile, os
import subprocess

import config

def run_matlab_code(code, verbose=False):
    # Write to a temp script
    (fd1, script_file) = tempfile.mkstemp(suffix='.m')
    (fd2, stdout_file) = tempfile.mkstemp(suffix='.txt')
    (fd3, stderr_file) = tempfile.mkstemp(suffix='.txt')
    
    f = open(script_file, 'w')
    f.write(code)
    f.close()
    
    call = [config.MATLAB_LOCATION, '-nosplash', '-nojvm', '-nodisplay']
    print call
    
    stdin = open(script_file)
    stdout = open(stdout_file, 'w')
    stderr = open(stderr_file, 'w')
    
#    subprocess.call(call, stdin=open(script_file), stdout=open(stdout_file, 'w'), stderr=open(stderr_file, 'w'))
    subprocess.call(call, stdin=stdin, stdout=stdout, stderr=stderr)
    
    stdin.close()
    stdout.close()
    stderr.close()
    
    f = open(stderr_file)
    err_txt = f.read()
    f.close()
    
    os.close(fd1)
    os.close(fd2)
    os.close(fd3)    
    
    if verbose:
        print
        print 'Script file (%s) contents : ==========================================' % script_file
        print open(script_file, 'r').read()
        print
        print 'Std out : =========================================='        
        print open(stdout_file, 'r').read()   
    
    if err_txt != '':
        #### TODO - need to catch error local to new MLG machines
#        print 'Matlab produced the following errors:\n\n%s' % err_txt    
        pass 
#        raise RuntimeError('Matlab produced the following errors:\n\n%s' % err_txt)
    else:     
        # Only remove temporary files if run was successful    
        os.remove(script_file)
        os.remove(stdout_file)
        os.remove(stderr_file)
    


# Matlab code to optimise hyper-parameters on one file, given one kernel.
OPTIMIZE_KERNEL_CODE = r"""
a='Load the data, it should contain X and y.'
load '%(datafile)s'

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = {@meanConst}
hyp.mean = mean(y)

covfunc = %(kernel_family)s
hyp.cov = %(kernel_params)s

likfunc = @likGauss
hyp.lik = %(noise)s

[hyp_opt, nlls] = minimize(hyp, @gp, -%(iters)s, @infExact, meanfunc, covfunc, likfunc, X, y);
best_nll = nlls(end)

%% Compute Hessian numerically for laplace approx
num_hypers = length(hyp.cov);
hessian = NaN(num_hypers, num_hypers);
delta = 1e-6;
a='Get original gradients';
[nll_orig, dnll_orig] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
for d = 1:num_hypers
    dhyp_opt = hyp_opt;
    dhyp_opt.cov(d) = dhyp_opt.cov(d) + delta;
    [nll_delta, dnll_delta] = gp(dhyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
    hessian(d, :) = (dnll_delta.cov - dnll_orig.cov) ./ delta;
end
hessian = 0.5 * (hessian + hessian');

save( '%(writefile)s', 'hyp_opt', 'best_nll', 'nlls', 'hessian' );
exit();
"""

class OptimizerOutput:
    def __init__(self, kernel_hypers, nll, nlls, hessian):
        self.kernel_hypers = kernel_hypers
        self.nll = nll
        self.nlls = nlls
        self.hessian = hessian

def optimize_params(kernel_expression, kernel_init_params, X, y, return_all=False, verbose=False, noise=None, iters=300):
    if X.ndim == 1:
        X = X[:, nax]
    if y.ndim == 1:
        y = y[:, nax]
        
    if noise is None:
        noise = np.log(np.var(y)/10)   # Just a heuristic.
        
    data = {'X': X, 'y': y}
    (fd1, temp_data_file) = tempfile.mkstemp(suffix='.mat')
    (fd2, temp_write_file) = tempfile.mkstemp(suffix='.mat')
    scipy.io.savemat(temp_data_file, data)
    
    if verbose:
        print kernel_init_params
    
    code = OPTIMIZE_KERNEL_CODE % {'datafile': temp_data_file,
                                   'writefile': temp_write_file,
                                   'gpml_path': config.GPML_PATH,
                                   'kernel_family': kernel_expression,
                                   'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel_init_params),
                                   'noise': str(noise),
                                   'iters': str(iters)}
    run_matlab_code(code, verbose)

    # Load in the file that GPML saved things to.
    gpml_result = scipy.io.loadmat(temp_write_file)
    
    os.close(fd1)
    os.close(fd2)
    os.remove(temp_data_file)
    os.remove(temp_write_file)

    optimized_hypers = gpml_result['hyp_opt']
    nll = gpml_result['best_nll'][0, 0]
    nlls = gpml_result['nlls'].ravel()
    hessian = gpml_result['hessian']

    # Strip out only kernel hyper-parameters.
    kernel_hypers = optimized_hypers['cov'][0, 0].ravel()

    return OptimizerOutput(kernel_hypers, nll, nlls, hessian)


# Some Matlab code to sample from a GP prior, in a spectral way.
GENERATE_NOISELESS_DATA_CODE = r"""
%% Load the data, it should contain X
load '%(datafile)s'

addpath(genpath('%(gpml_path)s'));

covfunc = %(kernel_family)s
hypers = %(kernel_params)s

sigma = feval(covfunc{:}, hypers, X);  
sigma = 0.5.*(sigma + sigma');
[vectors, values] = eig(sigma);
values(values < 0) = 0;
sample = vectors*(randn(length(values), 1).*sqrt(diag(values)));

save( '%(writefile)s', 'sample' );
exit();
"""

def sample_from_gp_prior(kernel, X):

    data = {'X': X}
    (fd1, temp_data_file) = tempfile.mkstemp(suffix='.mat')
    (fd2, temp_write_file) = tempfile.mkstemp(suffix='.mat')
    scipy.io.savemat(temp_data_file, data)
    
    kernel_params = kernel.param_vector()

    code = GENERATE_NOISELESS_DATA_CODE % {'datafile': temp_data_file,
                                   'writefile': temp_write_file,
                                   'gpml_path': config.GPML_PATH,
                                   'kernel_family': kernel.gpml_kernel_expression(),
                                   'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel_params)}
    run_matlab_code(code, verbose=True)

    # Load in the file that GPML saved things to.
    gpml_result = scipy.io.loadmat(temp_write_file)
    
    os.close(fd1)
    os.close(fd2)
    os.remove(temp_data_file)
    os.remove(temp_write_file)

    sample = gpml_result['sample'].ravel()
    return sample


# Matlab code to evaluate a covariance function at a bunch of locations.
EVAL_KERNEL_CODE = r"""
%% Load the data, it should contain X and x0.
load '%(datafile)s'

a='Load GPML'
addpath(genpath('%(gpml_path)s'));

covfunc = %(kernel_family)s
hypers = %(kernel_params)s

sigma = feval(covfunc{:}, hypers, X, x0);

save( '%(writefile)s', 'sigma' );
exit();
"""


def plot_kernel(kernel, X):

    data = {'X': X, 'x0': 0.0}
    (fd1, temp_data_file) = tempfile.mkstemp(suffix='.mat')
    (fd2, temp_write_file) = tempfile.mkstemp(suffix='.mat')
    scipy.io.savemat(temp_data_file, data)
    
    kernel_params = kernel.param_vector()

    code = EVAL_KERNEL_CODE % {'datafile': temp_data_file,
                               'writefile': temp_write_file,
                               'gpml_path': config.GPML_PATH,
                               'kernel_family': kernel.gpml_kernel_expression(),
                               'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel_params)}
    run_matlab_code(code, verbose=True)

    # Load in the file that GPML saved things to.
    gpml_result = scipy.io.loadmat(temp_write_file)
    
    os.close(fd1)
    os.close(fd2)
    os.remove(temp_data_file)
    os.remove(temp_write_file)

    return gpml_result['sigma'].ravel()

# Matlab code to compute mean fit
MEAN_FUNCTION_CODE = r"""
%% Load the data, it should contain X, y, X_test
load '%(datafile)s'

addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = {@meanConst}
hyp.mean = mean(y)

covfunc = %(kernel_family)s
hyp.cov = %(kernel_params)s

likfunc = @likGauss
hyp.lik = %(noise)s

[hyp, nlls] = minimize(hyp, @gp, -%(iters)s, @infExact, meanfunc, covfunc, likfunc, X, y);

%%HACK

hyp.cov = %(kernel_params)s
K = feval(covfunc{:}, hyp.cov, X);
K = K + exp(hyp.lik * 2) * eye(size(K));

%% We have now found appropriate mean and noise parameters

component_covfunc = %(component_kernel_family)s
hyp.cov = %(component_kernel_params)s

component_K = feval(component_covfunc{:}, hyp.cov, X, X_test)';

posterior_mean = component_K * (K \ y);

save( '%(writefile)s', 'posterior_mean' );
exit();
"""


def posterior_mean (kernel, component_kernel, X, y, X_test=None, noise=None, iters=300):
    #### Problem - we are not storing the learnt mean and noise - will need to re-learn - might not be especially correct!
    #### This is therefore just a placeholder
    if X.ndim == 1:
        X = X[:, nax]
    if y.ndim == 1:
        y = y[:, nax]
        
    if noise is None:
        noise = np.log(np.var(y)/10)   # Just a heuristic.
        
    if X_test is None:
        X_test = np.copy(X)
        
    data = {'X': X, 'y': y, 'X_test' : X_test}
    (fd1, temp_data_file) = tempfile.mkstemp(suffix='.mat')
    (fd2, temp_write_file) = tempfile.mkstemp(suffix='.mat')
    scipy.io.savemat(temp_data_file, data)
    
    code = MEAN_FUNCTION_CODE % {'datafile': temp_data_file,
                                 'writefile': temp_write_file,
                                 'gpml_path': config.GPML_PATH,
                                 'kernel_family': kernel.gpml_kernel_expression(),
                                 'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel.param_vector()),
                                 'component_kernel_family': component_kernel.gpml_kernel_expression(),
                                 'component_kernel_params': '[ %s ]' % ' '.join(str(p) for p in component_kernel.param_vector()),
                                 'noise': str(noise),
                                 'iters': str(iters)}
    
    run_matlab_code(code)

    # Load in the file that GPML saved things to.
    gpml_result = scipy.io.loadmat(temp_write_file)
    
    os.close(fd1)
    os.close(fd2)
    os.remove(temp_data_file)
    os.remove(temp_write_file)

    return gpml_result['posterior_mean'].ravel()


# Matlab code to make predictions on a dataset.
PREDICT_AND_SAVE_CODE = r"""
a='Load the data, it should contain X and y.'
load '%(datafile)s'

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = {@meanConst}
hyp.mean = mean(y)

covfunc = %(kernel_family)s
hyp.cov = %(kernel_params)s

likfunc = @likGauss
hyp.lik = %(noise)s

%% Optimize a little anyways.
[hyp_opt, nlls] = minimize(hyp, @gp, -%(iters)s, @infExact, meanfunc, covfunc, likfunc, X, y);
best_nll = nlls(end)

model.hypers = hyp_opt;

%% Evaluate a test points.
[ymu, ys2, predictions, fs2, loglik] = gp(model.hypers, @infExact, meanfunc, covfunc, likfunc, X, y, Xtest, ytest)

actuals = ytest;
train_time = NaN;
trainfolds = NaN;
testfolds = NaN;
K = NaN;
fold = NaN;
seed = NaN;
outdir = NaN;
timestamp = now

'%(writefile)s'

%% Save all the results.        
%%save( '%(writefile)s', 'loglik', 'predictions', 'actuals', 'model', 'train_time', 'trainfolds', 'testfolds', 'K', 'fold', 'seed', 'outdir', 'timestamp' );
save( '%(writefile)s', 'loglik', 'predictions', 'actuals', 'model', 'timestamp' );
exit();
"""


def make_predictions(kernel_expression, kernel_init_params, data_file, write_file, noise, iters=30):        
    code = PREDICT_AND_SAVE_CODE % {'datafile': data_file,
                                    'writefile': write_file,
                                    'gpml_path': config.GPML_PATH,
                                    'kernel_family': kernel_expression,
                                    'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel_init_params),
                                    'noise': str(noise),
                                    'iters': str(iters)}
    run_matlab_code(code, verbose=True)

