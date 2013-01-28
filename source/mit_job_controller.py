import numpy as np
nax = np.newaxis
import os
import scipy.io

import config
import flexiblekernel
import gpml
import mitparallel as mp



def evaluate_kernel_code(kernel, data_file, output_file, noise, iters):
    kernel_params = '[ ' + ' '.join(map(str, kernel.param_vector())) + ' ]'
    return gpml.OPTIMIZE_KERNEL_CODE % {'datafile': data_file,
                                        'writefile': output_file,
                                        'gpml_path': config.GPML_PATH,
                                        'kernel_family': kernel.gpml_kernel_expression(),
                                        'kernel_params': kernel_params,
                                        'noise': str(noise),
                                        'iters': str(iters),
                                        }

    
    
def evaluate_kernel(kernel, X, y, noise=None, iters=300):
    '''
    Sets up a kernel optimisation and nll calculation experiment, returns the result as scored kernel
    Input:
     - kernel            - A kernel (i.e. not scored kernel)
     - X                 - A matrix (data_points x dimensions) of input locations
     - y                 - A matrix (data_points x 1) of output values
     - ...
    Return:
     - A ScoredKernel object
    '''

    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    ndata = y.shape[0]
        
    # Set default noise using a heuristic.    
    if noise is None:
        noise = np.log(np.var(y)/10)
    
    data_file = mp.util.create_temp_file('.mat')
    scipy.io.savemat(data_file, {'X': X, 'y': y}) # Save regression data
    output_file = mp.util.create_temp_file('.mat')

    script = evaluate_kernel_code(kernel, data_file, output_file, noise, iters)
    mp.matlab.run(script)

    result = flexiblekernel.ScoredKernel.from_matlab_output(gpml.read_outputs(output_file),
                                                            kernel.family(), ndata)

    os.remove(data_file)
    os.remove(output_file)

    return result


def make_predictions_code(kernel, data_file, output_file):
    kernel_params = '[ ' + ' '.join(map(str, kernel.param_vector())) + ' ]'
    return gpml.PREDICT_AND_SAVE_CODE % {'datafile': data_file,
                                         'writefile': output_file,
                                         'gpml_path': config.GPML_PATH,
                                         'kernel_family': kernel.k_opt.gpml_kernel_expression(),
                                         'kernel_params': kernel_params,
                                         'noise': str(kernel.noise),
                                         'iters': str(30),
                                         }


def make_predictions(kernel, X, y, Xtest, ytest):
    """
    Evaluates a kernel on held out data
    Input:
     - X                  - A matrix (data_points x dimensions) of input locations
     - y                  - A matrix (data_points x 1) of output values
     - Xtest              - Held out X data
     - ytest              - Held out y data
     - best_scored_kernel - A Scored Kernel object to be evaluated on the held out data
     - ...
    Return:
     - A dictionary of results from the MATLAB script containing:
       - loglik - an array of log likelihoods of test data
       - predictions - an array of mean predictions for the held out data
       - actuals - ytest
       - model - I'm not sure FIXME
       - timestamp - A time stamp of some sort
    """
    
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    if Xtest.ndim == 1: Xtest = Xtest[:, nax]
    if ytest.ndim == 1: ytest = ytest[:, nax]

    data_file = mp.util.create_temp_file('.mat')
    scipy.io.savemat(data_file, {'X': X, 'y': y, 'Xtest' : Xtest, 'ytest' : ytest})
    output_file = mp.util.create_temp_file('.mat')

    code = make_predictions_code(kernel, data_file, output_file)
    print code
    assert False  # haven't tested yet, make sure code is reasonable
    mp.matlab.run(code)

    results = scipy.io.loadmat(output_file)

    os.remove(data_file)
    os.remove(output_file)

    return results

def compute_K_code(kernels, data_file, output_file, randproj):
    header = gpml.COMPUTE_K_CODE_HEADER % {'datafile': data_file,
                                           'gpml_path': config.GPML_PATH,
                                           }
    
    body = ''
    for i, kernel in enumerate(kernels):
        k_opt = kernel.k_opt
        kernel_family = k_opt.gpml_kernel_expression()
        kernel_params = '[ ' +  ' '.join(map(str, k_opt.param_vector())) + ' ]'
        body += gpml.SIMILARITY_CODE_COV % {'iter': i + 1,
                                            'kernel_family': kernel_family,
                                            'kernel_params': kernel_params}

    footer = gpml.COMPUTE_K_CODE_FOOTER % {'randproj': str(randproj).lower(),
                                           'writefile': output_file,
                                           }

    return '\n'.join([header, body, footer])


def compute_K(kernels, X, Q=None):
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]

    # random projections
    randproj = (Q is not None)

    data_file = mp.util.create_temp_file('.mat')
    if randproj:
        scipy.io.savemat(data_file, {'X': X, 'Q': Q})
    else:
        scipy.io.savemat(data_file, {'X': X})
    output_file = mp.util.create_temp_file('.mat')

    code = compute_K_code(kernels, data_file, output_file, randproj)
    mp.matlab.run(code)

    results = scipy.io.loadmat(output_file)
    K_list = [results['cov_matrices'][0, i] for i in range(results['cov_matrices'].shape[1])]

    os.remove(data_file)
    os.remove(output_file)

    return K_list


