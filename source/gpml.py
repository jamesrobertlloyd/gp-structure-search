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
import flexiblekernel as fk


def run_matlab_code(code, verbose=False, jvm=True):
    # Write to a temp script
    (fd1, script_file) = tempfile.mkstemp(suffix='.m')
    (fd2, stdout_file) = tempfile.mkstemp(suffix='.txt')
    (fd3, stderr_file) = tempfile.mkstemp(suffix='.txt')
    
    f = open(script_file, 'w')
    f.write(code)
    f.close()
    
    jvm_string = '-nojvm'
    if jvm: jvm_string = ''
    call = [config.MATLAB_LOCATION, '-nosplash', jvm_string, '-nodisplay']
    print call
    
    stdin = open(script_file)
    stdout = open(stdout_file, 'w')
    stderr = open(stderr_file, 'w')
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
rand('twister', %(seed)s);
randn('state', %(seed)s);

a='Load the data, it should contain X and y.'
load '%(datafile)s'
X = double(X)
y = double(y)

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = {@meanConst}
hyp.mean = mean(y)

covfunc = %(kernel_family)s
hyp.cov = %(kernel_params)s

likfunc = @likGauss
hyp.lik = %(noise)s

%% Repeat...
[hyp_opt, nlls] = minimize(hyp, @gp, -int32(%(iters)s * 3 / 3), @infExact, meanfunc, covfunc, likfunc, X, y);
%% ...optimisation - hopefully restarting optimiser will make it more robust to scale issues
%% [hyp_opt, nlls_2] = minimize(hyp_opt, @gp, -int32(%(iters)s * 3 / 3), @infExact, meanfunc, covfunc, likfunc, X, y);
%% nlls = [nlls_1; nlls_2];
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
%% exit();
"""

OPTIMIZE_KERNEL_CODE_ZERO_MEAN = r"""
rand('twister', %(seed)s);
randn('state', %(seed)s);

a='Load the data, it should contain X and y.'
load '%(datafile)s'
X = double(X)
y = double(y)

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = {@meanZero}
hyp.mean = [];

covfunc = %(kernel_family)s
hyp.cov = %(kernel_params)s

likfunc = @likGauss
hyp.lik = %(noise)s

%% Repeat...
[hyp_opt, nlls] = minimize(hyp, @gp, -int32(%(iters)s * 3 / 3), @infExact, meanfunc, covfunc, likfunc, X, y);
%% ...optimisation - hopefully restarting optimiser will make it more robust to scale issues
%% [hyp_opt, nlls_2] = minimize(hyp_opt, @gp, -int32(%(iters)s * 3 / 3), @infExact, meanfunc, covfunc, likfunc, X, y);
%% nlls = [nlls_1; nlls_2];
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
%% exit();
"""

class OptimizerOutput:
    def __init__(self, kernel_hypers, nll, nlls, hessian, noise_hyp):
        self.kernel_hypers = kernel_hypers
        self.nll = nll
        self.nlls = nlls
        self.hessian = hessian
        self.noise_hyp = noise_hyp

def optimize_params(kernel_expression, kernel_init_params, X, y, return_all=False, verbose=False, noise=None, iters=300, zero_mean=False, random_seed=0):
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
    
    parameters =  {'datafile': temp_data_file,
                   'writefile': temp_write_file,
                   'gpml_path': config.GPML_PATH,
                   'kernel_family': kernel_expression,
                   'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel_init_params),
                   'noise': str(noise),
                   'iters': str(iters),
                   'seed': str(random_seed)}
    if zero_mean:
        code = OPTIMIZE_KERNEL_CODE_ZERO_MEAN % parameters
    else:
        code = OPTIMIZE_KERNEL_CODE % parameters
    run_matlab_code(code, verbose)
    
    output = read_outputs(temp_write_file)
    
    os.close(fd1)
    os.close(fd2)
    os.remove(temp_data_file)
    os.remove(temp_write_file)

    return output


def read_outputs(write_file):
    gpml_result = scipy.io.loadmat(write_file)
    optimized_hypers = gpml_result['hyp_opt']
    nll = gpml_result['best_nll'][0, 0]
    hessian = gpml_result['hessian']
    nlls = gpml_result['nlls'].ravel()
    assert isinstance(hessian, np.ndarray)  # just to make sure

    kernel_hypers = optimized_hypers['cov'][0, 0].ravel()
    noise_hyp = optimized_hypers['lik'][0, 0].ravel()
    
    return OptimizerOutput(kernel_hypers, nll, nlls, hessian, noise_hyp)
                        


# Some Matlab code to sample from a GP prior, in a spectral way.
GENERATE_NOISELESS_DATA_CODE = r"""
%% Load the data, it should contain X
load '%(datafile)s'
X = double(X)

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
X = double(X)
x0 = double(x0)

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
rand('twister', %(seed)s);
randn('state', %(seed)s);

%% Load the data, it should contain X, y, X_test
load '%(datafile)s'
X = double(X)
y = double(y)
Xtest = double(Xtest)

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

MEAN_FUNCTION_CODE_ZERO_MEAN = r"""
rand('twister', %(seed)s);
randn('state', %(seed)s);

%% Load the data, it should contain X, y, X_test
load '%(datafile)s'
X = double(X)
y = double(y)
Xtest = double(Xtest)

addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = {@meanZero}
hyp.mean = []

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


def posterior_mean (kernel, component_kernel, X, y, X_test=None, noise=None, iters=300, zero_mean=False, random_seed=0):
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
    
    parameters ={'datafile': temp_data_file,
                 'writefile': temp_write_file,
                 'gpml_path': config.GPML_PATH,
                 'kernel_family': kernel.gpml_kernel_expression(),
                 'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel.param_vector()),
                 'component_kernel_family': component_kernel.gpml_kernel_expression(),
                 'component_kernel_params': '[ %s ]' % ' '.join(str(p) for p in component_kernel.param_vector()),
                 'noise': str(noise),
                 'iters': str(iters),
                 'seed': str(random_seed)}
    
    if zero_mean:
        code = MEAN_FUNCTION_CODE_ZERO_MEAN % parameters
    else:
        code = MEAN_FUNCTION_CODE % parameters
    
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
rand('twister', %(seed)s);
randn('state', %(seed)s);

a='Load the data, it should contain X and y.'
load '%(datafile)s'
X = double(X)
y = double(y)
Xtest = double(Xtest)
ytest = double(ytest)

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
timestamp = now

'%(writefile)s'

save('%(writefile)s', 'loglik', 'predictions', 'actuals', 'model', 'timestamp');

pwd
%% save('/home/dkd23/Dropbox/results/r_pumadyn512_fold_3_of_10_result.txt'

a='Supposedly finished writing file'

%% exit();
"""

PREDICT_AND_SAVE_CODE_ZERO_MEAN = r"""
rand('twister', %(seed)s);
randn('state', %(seed)s);

a='Load the data, it should contain X and y.'
load '%(datafile)s'
X = double(X)
y = double(y)
Xtest = double(Xtest)
ytest = double(ytest)

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = {@meanZero}
hyp.mean = []

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
timestamp = now

'%(writefile)s'

save('%(writefile)s', 'loglik', 'predictions', 'actuals', 'model', 'timestamp');

pwd
%% save('/home/dkd23/Dropbox/results/r_pumadyn512_fold_3_of_10_result.txt'

a='Supposedly finished writing file'

%% exit();
"""

#### TODO - remove me
def make_predictions(kernel_expression, kernel_init_params, data_file, write_file, noise, iters=30, zero_mean=False, random_seed=0):  
    parameters = {'datafile': data_file,
                  'writefile': write_file,
                  'gpml_path': config.GPML_PATH,
                  'kernel_family': kernel_expression,
                  'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel_init_params),
                  'noise': str(noise),
                  'iters': str(iters),
                  'seed': str(random_seed)}
    if zero_mean:
        code = PREDICT_AND_SAVE_CODE_ZERO_MEAN % parameters
    else:
        code = PREDICT_AND_SAVE_CODE % parameters
    run_matlab_code(code, verbose=True)
    
# Matlab code to evaluate DISTANCE of kernels
DISTANCE_CODE_HEADER = r"""
fprintf('Load the data, it should contain inputs X')
load '%(datafile)s'
X = double(X)

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Create list of covariance functions
"""

DISTANCE_CODE_COV = r"""
covs{%(iter)d} = %(kernel_family)s
hyps{%(iter)d} = %(kernel_params)s
"""

DISTANCE_CODE_FOOTER = r"""
%% Evaluate similarities
n_kernels = length(covs);
sim_matrix = zeros(n_kernels);
%% Note - quadratic evaluations of kernels to avoid huge memory requirements
for i = 1:n_kernels
  cov_i = feval(covs{i}{:}, hyps{i}, X);
  for j = (i+1):n_kernels
    cov_j = feval(covs{j}{:}, hyps{j}, X);
    %% Compute Frobenius norm
    sq_diff = (cov_i - cov_j) .^ 2;
    frobenius = sqrt(sum(sq_diff(:)));
    %% Put in sim matrix
    sim_matrix(i, j) = frobenius;
  end
end
%% Make symmetric
sim_matrix = sim_matrix + sim_matrix';

save( '%(writefile)s', 'sim_matrix' );
"""

DISTANCE_CODE_FOOTER_HIGH_MEM = r"""
%% Precompute covariance matrices
for i = 1:length(covs);
  cov_matrices{i} = feval(covs{i}{:}, hyps{i}, X);
end
%% Evaluate similarities
n_kernels = length(covs);
sim_matrix = zeros(n_kernels);
for i = 1:n_kernels
  for j = (i+1):n_kernels
    %% Compute Frobenius norm
    sq_diff = (cov_matrices{i} - cov_matrices{j}) .^ 2;
    frobenius = sqrt(sum(sq_diff(:)));
    %% Put in sim matrix
    sim_matrix(i, j) = frobenius;
  end
end
%% Make symmetric
sim_matrix = sim_matrix + sim_matrix';

save( '%(writefile)s', 'sim_matrix' );
"""


# Matlab code to decompose posterior into additive parts.
MATLAB_PLOT_DECOMP_CALLER_CODE = r"""
load '%(datafile)s'  %% Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('%(gpml_path)s'));
addpath(genpath('%(matlab_script_path)s'));

plot_decomp(X, y, %(kernel_family)s, %(kernel_params)s, %(kernel_family_list)s, %(kernel_params_list)s, %(noise)s, '%(figname)s', %(latex_names)s, %(full_kernel_name)s, %(X_mean)f, %(X_scale)f, %(y_mean)f, %(y_scale)f)
exit();"""


def plot_decomposition(kernel, X, y, figname, noise=None, X_mean=0, X_scale=1, y_mean=0, y_scale=1):
    matlab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'matlab'))
    figname = os.path.abspath(os.path.join(os.path.dirname(__file__), figname))
    print 'Plotting to: %s' % figname
    
    #### TODO - make break_kernel... do something
    kernel_components = fk.break_kernel_into_summands(kernel)
    latex_names = [k.latex_print().strip() for k in kernel_components]
    kernel_params_list = ','.join('[ %s ]' % ' '.join(str(p) for p in k.param_vector()) for k in kernel_components)
    
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    if noise is None: noise = np.log(np.var(y)/10)   # Just a heuristic.
    data = {'X': X, 'y': y}
    (fd1, temp_data_file) = tempfile.mkstemp(suffix='.mat')
    scipy.io.savemat(temp_data_file, data)
    
    code = MATLAB_PLOT_DECOMP_CALLER_CODE % {'datafile': temp_data_file,
        'gpml_path': config.GPML_PATH,
        'matlab_script_path': matlab_dir,
        'kernel_family': kernel.gpml_kernel_expression(),
        'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel.param_vector()),
        'kernel_family_list': '{ %s }' % ','.join(str(k.gpml_kernel_expression()) for k in kernel_components),
        'kernel_params_list': '{ %s }' % kernel_params_list,
        'noise': str(noise),
        'latex_names': "{ ' %s ' }" % "','".join(latex_names),
        'full_kernel_name': "{ '%s' }" % kernel.latex_print().strip(), 
        'figname': figname,
        'X_mean' : X_mean,
        'X_scale' : X_scale,
        'y_mean' : y_mean,
        'y_scale' : y_scale}
    
    run_matlab_code(code, verbose=True, jvm=True)
    os.close(fd1)
    #os.remove(temp_data_file)


def load_mat(data_file, y_dim=1):
    '''
    Load a Matlab file containing inputs X and outputs y, output as np.arrays
     - X is (data points) x (input dimensions) array
     - y is (data points) x (output dimensions) array
     - y_dim selects which output dimension is returned (1 indexed)
    Returns tuple (X, y, # data points)
    '''
     
    data = scipy.io.loadmat(data_file)
    #### TODO - this should return a dictionary, not a tuple
    if 'Xtest' in data:
        return data['X'], data['y'][:,y_dim-1], np.shape(data['X'])[1], data['Xtest'], data['ytest'][:,y_dim-1]
    else:
        return data['X'], data['y'][:,y_dim-1], np.shape(data['X'])[1]


COMPUTE_K_CODE_HEADER = r"""
fprintf('Load the data, it should contain inputs X');
load '%(datafile)s';
X = double(X)

%% Load GPML
addpath(genpath('%(gpml_path)s'));

%% Create list of covariance functions"""

COMPUTE_K_CODE_COVS = r"""
covs{%(iter)d} = %(kernel_family)s;
hyps{%(iter)d} = %(kernel_params)s;
"""

COMPUTE_K_CODE_FOOTER = r"""
%% Random projection
randproj = %(randproj)s;

for i = 1:length(covs)
  K = feval(covs{i}{:}, hyps{i}, X);

  if randproj
    K = K(:);
    K = Q' * K;
  end

  cov_matrices{i} = K;
end

save('%(writefile)s', 'cov_matrices');
"""

