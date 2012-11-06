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
    script_file = tempfile.mkstemp(suffix='.m')[1]
    stdout_file = tempfile.mkstemp(suffix='.txt')[1]
    stderr_file = tempfile.mkstemp(suffix='.txt')[1]
    
    f = open(script_file, 'w')
    f.write(code)
    f.close()
    
    call = [config.MATLAB_LOCATION, '-nosplash', '-nojvm', '-nodisplay']
    subprocess.call(call, stdin=open(script_file), stdout=open(stdout_file, 'w'), stderr=open(stderr_file, 'w'))
    
    f = open(stderr_file)
    err_txt = f.read()
    f.close()
    
    if err_txt != '':
        #### TODO - need to catch error local to new MLG machines
#        print 'Matlab produced the following errors:\n\n%s' % err_txt
        if verbose:
            print
            print 'Script file (%s) contents : ==========================================' % script_file
            print open(script_file, 'r').read()
            print
            print 'Std out : =========================================='        
            print open(stdout_file, 'r').read()        
        raise RuntimeError('Matlab produced the following errors:\n\n%s' % err_txt)
    else:     
        # Only remove temporary files if run was successful    
        os.remove(script_file)
        os.remove(stdout_file)
        os.remove(stderr_file)
    


# Matlab code to optimise hyper-parameters on one file, given one kernel.
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
hyp.lik = %(noise)s

[hyp_opt, nlls] = minimize(hyp, @gp, -%(iters)s, @infExact, meanfunc, covfunc, likfunc, X, y);
best_nll = nlls(end)

save( '%(writefile)s', 'hyp_opt', 'best_nll', 'nlls' );
exit();
"""

def optimize_params(kernel_expression, kernel_init_params, X, y, return_all=False, verbose=False, noise=None, iters=300):
    if X.ndim == 1:
        X = X[:, nax]
    if y.ndim == 1:
        y = y[:, nax]
        
    if noise is None:
        noise = np.log(np.var(y, 1)/10)
        
    data = {'X': X, 'y': y}
    temp_data_file = tempfile.mkstemp(suffix='.mat')[1]
    temp_write_file = tempfile.mkstemp(suffix='.mat')[1]
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
    run_matlab_code(code)

    # Load in the file that GPML saved things to.
    gpml_result = scipy.io.loadmat(temp_write_file)
    os.remove(temp_data_file)
    os.remove(temp_write_file)

    optimized_hypers = gpml_result['hyp_opt']
    nll = gpml_result['best_nll'][0, 0]
    nlls = gpml_result['nlls'].ravel()

    # Strip out only kernel hyper-parameters.
    kernel_hypers = optimized_hypers['cov'][0, 0].ravel()

    if return_all:
        return kernel_hypers, nll, nlls
    else:
        return kernel_hypers, nll


# Some Matlab code to sample from a GP prior, in a spectral way.
GENERATE_NOISELESS_DATA_CODE = r"""
a='Load the data, it should contain X'
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
    temp_data_file = tempfile.mkstemp(suffix='.mat')[1]
    temp_write_file = tempfile.mkstemp(suffix='.mat')[1]
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
    os.remove(temp_data_file)
    os.remove(temp_write_file)

    sample = gpml_result['sample'].ravel()
    return sample


# Matlab code to evaluate a covariance function at a bunch of locations.
EVAL_KERNEL_CODE = r"""
a='Load the data, it should contain X and x0.'
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
    temp_data_file = tempfile.mkstemp(suffix='.mat')[1]
    temp_write_file = tempfile.mkstemp(suffix='.mat')[1]
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
    os.remove(temp_data_file)
    os.remove(temp_write_file)

    return gpml_result['sigma'].ravel()



