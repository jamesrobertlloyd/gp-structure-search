'''
Main file for dispatching jobs to a cluster, creates remote files, etc.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
Created Jan 2013
'''

import flexiblekernel as fk
from flexiblekernel import ScoredKernel
import grammar
import gpml
import utils.latex
import utils.fear
from config import *
from utils import gaussians, psd_matrices

import numpy as np
nax = np.newaxis
import pylab
import scipy.io
import sys
import os
import tempfile
import subprocess
import time

import cblparallel
from cblparallel.util import mkstemp_safe
import re

import shutil
import random


def covariance_similarity(kernels, X, local_computation=True, verbose=True): 
    '''
    Evaluate a similarity matrix or kernels, in terms of their covariance matrix evaluated on training inputs
    Assumes kernels is a list of ScoredKernel objects
    '''
    # Construct data and send to fear if appropriate
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    data = {'X': X}
	#### TODO - Move if statetments to cblparallel
    if not local_computation:
        # If not in CBL need to communicate with fear via gate.eng.cam.ac.uk
        fear = cblparallel.fear(via_gate=(LOCATION=='home'))
    if LOCATION=='home':
        data_file = mkstemp_safe(cblparallel.HOME_TEMP_PATH, '.mat')
    else:
        data_file = mkstemp_safe(cblparallel.LOCAL_TEMP_PATH, '.mat')
    scipy.io.savemat(data_file, data)
    if not local_computation:
        if verbose:
            print 'Moving data file to fear'
        fear.copy_to_temp(data_file)
    # Construct testing code
    if not local_computation:
        gpml_path = REMOTE_GPML_PATH
    elif LOCATION == 'local':
        gpml_path = LOCAL_GPML_PATH
    else:
        gpml_path = HOME_GPML_PATH
    code = gpml.SIMILARITY_CODE_HEADER % {'datafile': data_file.split('/')[-1],
                                          'gpml_path': gpml_path}
    for (i, kernel) in enumerate([k.k_opt for k in kernels]):
        code = code + gpml.SIMILARITY_CODE_COV % {'iter' : i + 1,
                                                  'kernel_family': kernel.gpml_kernel_expression(),
                                                  'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel.param_vector())}
    code = code + gpml.SIMILARITY_CODE_FOOTER_HIGH_MEM % {'writefile': '%(output_file)s'} # N.B. cblparallel manages output files
    code = re.sub('% ', '%% ', code) # HACK - cblparallel not fond of % signs
    # Run code
    if local_computation:
        output_file = cblparallel.run_batch_locally([code], language='matlab', max_cpu=1.1, max_mem=1.1, job_check_sleep=30, verbose=True, single_thread=False)[0] 
    else:
        output_file = cblparallel.run_batch_on_fear([code], language='matlab', max_jobs=500, verbose=verbose)[0]
    # Read in results
    gpml_result = scipy.io.loadmat(output_file)
    similarity = gpml_result['sim_matrix']
    # Tidy up
    os.remove(output_file)
    os.remove(data_file)
    if not local_computation:
        # TODO - hide paths from end user
        fear.rm(os.path.join(cblparallel.REMOTE_TEMP_PATH, os.path.split(data_file)[-1]))
        fear.disconnect()
    # Return
    return similarity

       
def evaluate_kernels(kernels, X, y, verbose=True, noise=None, iters=300, local_computation=False, zip_files=False, max_jobs=500):
    '''Sets up the experiments, sends them to cblparallel, returns the results.'''
   
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    data = {'X': X, 'y': y}
    ndata = y.shape[0]
        
    # Set default noise using a heuristic.    
    if noise is None:
        noise = np.log(np.var(y)/10)
    
    # Create a connection to fear if not performing calculations locally
    if not local_computation:
        # If not in CBL need to communicate with fear via gate.eng.cam.ac.uk
        fear = cblparallel.fear(via_gate=(LOCATION=='home'))
    
    # Create data file
    if verbose:
        print 'Creating data file locally'
    if LOCATION=='home':
        data_file = mkstemp_safe(cblparallel.HOME_TEMP_PATH, '.mat')
    else:
        data_file = mkstemp_safe(cblparallel.LOCAL_TEMP_PATH, '.mat')
    scipy.io.savemat(data_file, data) # Save regression data
    
    # Move to fear if necessary
    if not local_computation:
        if verbose:
            print 'Moving data file to fear'
        fear.copy_to_temp(data_file)
    
    # Create a list of MATLAB scripts to assess and optimise parameters for each kernel
    if verbose:
        print 'Creating scripts'
    scripts = [None] * len(kernels)
    if not local_computation:
        gpml_path = REMOTE_GPML_PATH
    elif LOCATION == 'local':
        gpml_path = LOCAL_GPML_PATH
    else:
        gpml_path = HOME_GPML_PATH
    for (i, kernel) in enumerate(kernels):
        scripts[i] = gpml.OPTIMIZE_KERNEL_CODE % {'datafile': data_file.split('/')[-1],
                                                  'writefile': '%(output_file)s', # N.B. cblparallel manages output files
                                                  'gpml_path': gpml_path,
                                                  'kernel_family': kernel.gpml_kernel_expression(),
                                                  'kernel_params': '[ %s ]' % ' '.join(str(p) for p in kernel.param_vector()),
                                                  'noise': str(noise),
                                                  'iters': str(iters)}
        #### Need to be careful with % signs
        #### For the moment, cblparallel expects no single % signs - FIXME
        scripts[i] = re.sub('% ', '%% ', scripts[i])
    
    # Send to cblparallel and save output_files
    if verbose:
        print 'Sending scripts to cblparallel'
    if local_computation:
        output_files = cblparallel.run_batch_locally(scripts, language='matlab', max_cpu=0.8, job_check_sleep=5, submit_sleep=0.1, max_running_jobs=6, verbose=verbose)  
    else:
        output_files = cblparallel.run_batch_on_fear(scripts, language='matlab', max_jobs=max_jobs, verbose=verbose, zip_files=zip_files)  
    
    # Read in results
    results = [None] * len(kernels)
    for (i, output_file) in enumerate(output_files):
        if verbose:
            print 'Reading output file %d of %d' % (i + 1, len(kernels))
        results[i] = ScoredKernel.from_matlab_output(gpml.read_outputs(output_file), kernels[i].family(), ndata)
    
    # Tidy up
    for (i, output_file) in enumerate(output_files):
        if verbose:
            print 'Removing output file %d of %d' % (i + 1, len(kernels)) 
        os.remove(output_file)
    os.remove(data_file)
    if not local_computation:
        # TODO - hide paths from end user
        fear.rm(os.path.join(cblparallel.REMOTE_TEMP_PATH, os.path.split(data_file)[-1]))
        fear.disconnect()
    
    # Return results
    return results     

   
def make_predictions(X, y, Xtest, ytest, best_scored_kernel, local_computation=False):
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    if y.ndim == 1: y = y[:, nax]
    data = {'X': X, 'y': y, 'Xtest' : Xtest, 'ytest' : ytest}
    ndata = y.shape[0]
    # Create a connection to fear if not performing calculations locally
    if LOCATION=='home':
        data_file = mkstemp_safe(cblparallel.HOME_TEMP_PATH, '.mat')
    else:
        data_file = mkstemp_safe(cblparallel.LOCAL_TEMP_PATH, '.mat')   
    scipy.io.savemat(data_file, data) # Save regression data
    if not local_computation:
        # If not in CBL need to communicate with fear via gate.eng.cam.ac.uk
        #if (LOCATION == 'home'):
        #    cblparallel.start_port_forwarding()
        fear = cblparallel.fear(via_gate=(LOCATION=='home'))
        # Move data file to fear
        #if verbose:
        #    print 'Moving data file to fear'
        fear.copy_to_temp(data_file)
    if not local_computation:
        gpml_path = REMOTE_GPML_PATH
    elif LOCATION == 'local':
        gpml_path = LOCAL_GPML_PATH
    else:
        gpml_path = HOME_GPML_PATH
    code = gpml.PREDICT_AND_SAVE_CODE % {'datafile': data_file.split('/')[-1],
                                         'writefile': '%(output_file)s',
                                         'gpml_path': gpml_path,
                                         'kernel_family': best_scored_kernel.k_opt.gpml_kernel_expression(),
                                         'kernel_params': '[ %s ]' % ' '.join(str(p) for p in best_scored_kernel.k_opt.param_vector()),
                                         'noise': str(best_scored_kernel.noise),
                                         'iters': str(30)}
    code = re.sub('% ', '%% ', code) # HACK - cblparallel currently does not like % signs
    if local_computation:   
        temp_results_file = cblparallel.run_batch_locally([code], language='matlab', max_cpu=1.1, max_mem=1.1)[0]
    else:
        temp_results_file = cblparallel.run_batch_on_fear([code], language='matlab')[0]
    results = scipy.io.loadmat(temp_results_file)
    os.remove(temp_results_file)
    if not local_computation:
        # TODO - hide paths from end user
        fear.rm(os.path.join(cblparallel.REMOTE_TEMP_PATH, os.path.split(data_file)[-1]))
        fear.disconnect()         
    elif LOCATION == 'home':
        os.remove(os.path.join(cblparallel.HOME_TEMP_PATH, os.path.split(data_file)[-1]))
    else:
        os.remove(os.path.join(cblparallel.LOCAL_TEMP_PATH, os.path.split(data_file)[-1]))   
    return results


