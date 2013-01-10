'''
Main file for performing structure search.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
Created November 2012          
'''

import flexiblekernel as fk
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
import itertools

import cblparallel
from cblparallel.util import mkstemp_safe
import re

import shutil

import random

PRIOR_VAR = 100.

def load_mat(data_file, y_dim=1):
    '''
    Load a Matlab file containing inputs X and outputs y, output as np.arrays
     - X is (data points) x (input dimensions) array
     - y is (data points) x (output dimensions) array
     - y_dim selects which output dimension is returned (1 indexed)
    Returns tuple (X, y, # data points)
    '''
     
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y'][:,y_dim-1], np.shape(data['X'])[1]

def proj_psd(H):
    '''
    Makes stuff psd I presume? Comments welcome.
    '''
    assert np.allclose(H, H.T), 'not symmetric'
    d, Q = scipy.linalg.eigh(H)
    d = np.clip(d, 1e-8, np.infty)
    return np.dot(Q, d[:, nax] * Q.T)
    

def laplace_approx(nll, opt_hyper, hessian, prior_var):
    #### FIXME - Believed to have a bug
    ####       - Might be MATLAB though - test this code on some known integrals
    d = opt_hyper.size
    
    hessian = proj_psd(hessian)

    # quadratic centered at opt_hyper with maximum -nll
    evidence = gaussians.Potential(np.zeros(d), psd_matrices.FullMatrix(hessian), -nll)
    evidence = evidence.translate(opt_hyper)

    # zero-centered Gaussian
    prior = gaussians.Potential.from_moments_iso(np.zeros(d), prior_var)

    # multiply the two Gaussians and integrate the result
    return -(evidence + prior).integral()


def expand_kernels(D, seed_kernels, verbose=False, debug=False):    
    '''Makes a list of all expansions of a set of kernels in D dimensions.'''
    g = grammar.MultiDGrammar(D, debug=debug)
    if verbose:
        print 'Seed kernels :'
        for k in seed_kernels:
            print k.pretty_print()
    kernels = []
    for k in seed_kernels:
        kernels = kernels + grammar.expand(k, g)
    kernels = grammar.remove_duplicates(kernels)
    if verbose:
        print 'Expanded kernels :'
        for k in kernels:
            print k.pretty_print()
    return (kernels)

def replace_defaults(param_vector, sd):
    #### FIXME - remove dependence on special value of zero
    ####       - Caution - remember print, compare etc when making the change (e.g. just replacing 0 with None would cause problems later)
    '''Replaces zeros in a list with Gaussians'''
    return [np.random.normal(scale=sd) if p ==0 else p for p in param_vector]

def add_random_restarts_single_kernel(kernel, n_rand, sd):
    '''Returns a list of kernels with random restarts for default values'''
    return [kernel] + list(itertools.repeat(kernel.family().from_param_vector(replace_defaults(kernel.param_vector(), sd)), n_rand))

def add_random_restarts(kernels, n_rand=1, sd=2):    
    '''Augments the list to include random restarts of all default value parameters'''
    return [k_rand for kernel in kernels for k_rand in add_random_restarts_single_kernel(kernel, n_rand, sd)]

class ScoredKernel:
    '''
    Wrapper around a kernel with various scores and noise parameter
    '''
    def __init__(self, k_opt, nll, laplace_nle, bic_nle, noise):
        self.k_opt = k_opt
        self.nll = nll
        self.laplace_nle = laplace_nle
        self.bic_nle = bic_nle
        self.noise = noise
        
    def score(self, criterion='bic'):
        #### FIXME - Change default to laplace when it is definitely bug free
        return {'bic': self.bic_nle,
                'nll': self.nll,
                'laplace': self.laplace_nle
                }[criterion]
                
    @staticmethod
    def from_printed_outputs(nll, laplace, BIC, noise=None, kernel=None):
        return ScoredKernel(kernel, nll, laplace, BIC, noise)
    
    def __repr__(self):
        return 'ScoredKernel(k_opt=%s, nll=%f, laplace_nle=%f, bic_nle=%f, noise=%s)' % \
            (self.k_opt, self.nll, self.laplace_nle, self.bic_nle, self.noise)
    
    @staticmethod
    def parse_results_string(line):
        #### FIXME - Higher level of python fu than I understand - can guess but can someone comment?
        v = locals().copy()
        v.update(fk.__dict__)
        v['nan'] = np.NaN;
        return eval(line, globals(), v) 
        
def covariance_similarity(kernels, X, local_computation=True, verbose=True): 
    '''
    Evaluate a similarity matrix or kernels, in terms of their covariance matrix evaluated on training inputs
    Assumes kernels is a list of ScoredKernel objects
    '''
    # Construct data and send to fear if appropriate
    # Make data into matrices in case they're unidimensional.
    if X.ndim == 1: X = X[:, nax]
    data = {'X': X}
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
        
def remove_duplicates(kernels, X, n_eval=250, local_computation=True):
    '''
    Test the top n_eval performing kernels for equivalence, in terms of their covariance matrix evaluated on training inputs
    Assumes kernels is a list of ScoredKernel objects
    '''
    #### HACK - this needs lots of computing power - do it locally with multi-threading
    local_computation = True
    # Sort
    kernels = sorted(kernels, key=ScoredKernel.score, reverse=False)
    # Find covariance similarity for top n_eval
    n_eval = min(n_eval, len(kernels))
    similarity_matrix = covariance_similarity(kernels[:n_eval], X, local_computation=local_computation)
    # Remove similar kernels
    #### TODO - What is a good heuristic for determining equivalence?
    ####      - Currently using Frobenius norm - truncate if Frobenius norm is below 1% of average
    cut_off = similarity_matrix.mean() / 100.0
    equivalence_graph = similarity_matrix < cut_off
    # For all kernels (best first)
    for i in range(n_eval):
        # For all other worse kernels
        for j in range(i+1, n_eval):
            # If equivalent
            if equivalence_graph[i,j]:
                # Destroy the inferior duplicate
                kernels[j] = None
    # Sort the results
    kernels = [k for k in kernels if k is not None]
    kernels = sorted(kernels, key=ScoredKernel.score, reverse=True)
    return kernels
        
def perform_kernel_search(data_file, results_filename, y_dim=1, subset=None, max_depth=2, k=2, \
                          verbose=True, description='No description', n_rand=1, sd=2, local_computation=False, debug=False, zip_files=False, max_jobs=500):
    '''Recursively search for the best kernel, in parallel on fear or local machine.'''

    # Load data - input, output and number of input dimensions
    X, y, D = load_mat(data_file, y_dim)
    # Initialise kernels to be base kernels
    if debug:
        current_kernels = list(fk.test_kernels(4))
    else:
        current_kernels = list(fk.base_kernels(D))
    # Initialise list of results
    results = []              # All results.
    results_sequence = []     # Results sets indexed by level of expansion.
    # Perform search
    for depth in range(max_depth):   
        # Add random restarts to kernels
        current_kernels = add_random_restarts(current_kernels, n_rand, sd)
        # Score the kernels
        new_results = evaluate_kernels(current_kernels, X, y, verbose=verbose, local_computation=local_computation, zip_files=zip_files, max_jobs=max_jobs)
        # Sort the new results
        new_results = sorted(new_results, key=ScoredKernel.score, reverse=True)
        # Remove near duplicates from these results (top m results only for efficiency)
        print 'Printing new results'
        for result in new_results:
            print result.nll, result.laplace_nle, result.bic_nle, result.k_opt.pretty_print()
        new_results = remove_duplicates(new_results, X, local_computation=local_computation)
        print 'Printing new results after duplicate removal'
        for result in new_results:
            print result.nll, result.laplace_nle, result.bic_nle, result.k_opt.pretty_print()
        # Collate results
        results = results + new_results
        # Sort all results
        results = sorted(results, key=ScoredKernel.score, reverse=True)
        # Remove duplicates - not necessary here? Heuristic can go a bit funny!
        # results = remove_duplicates(results, X, local_computation=local_computation)
        results_sequence.append(results)
        if verbose:
            print 'Printing all results'
            for result in results:
                print result.nll, result.laplace_nle, result.bic_nle, result.k_opt.pretty_print()
        # Extract the best k kernels from the new results
        #### Thoughts - Search can get stuck since can replace kernels in place
        ####          - Remove duplicates here - or higher up?
        best_kernels = [r.k_opt for r in sorted(new_results, key=ScoredKernel.score)[0:k]]
        if debug:
            current_kernels = expand_kernels(4, best_kernels, verbose=verbose, debug=debug)
        else:
            current_kernels = expand_kernels(D, best_kernels, verbose=verbose, debug=debug)

    # Write results to a file.
    results = sorted(results, key=ScoredKernel.score, reverse=True)
    with open(results_filename, 'w') as outfile:
        outfile.write('Experiment results for\n datafile = %s\n y_dim = %d\n subset = %s\n max_depth = %f\n k = %f\n Description = %s\n\n' \
                      % (data_file, y_dim, subset, max_depth, k, description)) 
        for (i, results) in enumerate(results_sequence):
            outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
            for result in results:
                print >> outfile, result  
           
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
        results[i] = output_to_scored_kernel(gpml.read_outputs(output_file), kernels[i].family(), ndata)
    
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
    
def make_predictions(data_file, results_file, prediction_file, local_computation=False):
    # Create a connection to fear if not performing calculations locally
    if not local_computation:
        # If not in CBL need to communicate with fear via gate.eng.cam.ac.uk
        #if (LOCATION == 'home'):
        #    cblparallel.start_port_forwarding()
        fear = cblparallel.fear(via_gate=(LOCATION=='home'))
        # Move data file to fear
        #if verbose:
        #    print 'Moving data file to fear'
        fear.copy_to_temp(data_file)
    elif LOCATION == 'home':
        # Save a copy of the data file in the temp directory
        shutil.copy(data_file, os.path.join(cblparallel.HOME_TEMP_PATH, os.path.split(data_file)[-1]))
    else:
        shutil.copy(data_file, os.path.join(cblparallel.LOCAL_TEMP_PATH, os.path.split(data_file)[-1]))
    best_scored_kernel = parse_results(results_file)
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
    # Move prediction file from temporary directory and tidy up
    shutil.copy(temp_results_file, prediction_file)
    os.remove(temp_results_file)
    if not local_computation:
        # TODO - hide paths from end user
        fear.rm(os.path.join(cblparallel.REMOTE_TEMP_PATH, os.path.split(data_file)[-1]))
        fear.disconnect()         
    elif LOCATION == 'home':
        os.remove(os.path.join(cblparallel.HOME_TEMP_PATH, os.path.split(data_file)[-1]))
    else:
        os.remove(os.path.join(cblparallel.LOCAL_TEMP_PATH, os.path.split(data_file)[-1]))   

def output_to_scored_kernel(output, kernel_family, ndata):
    '''Computes Laplace marginal lik approx and BIC - returns scored Kernel'''
    laplace_nle = laplace_approx(output.nll, output.kernel_hypers, output.hessian, PRIOR_VAR)
    k_opt = kernel_family.from_param_vector(output.kernel_hypers)
    BIC = 2 * output.nll + k_opt.effective_params() * np.log(ndata)
    return ScoredKernel(k_opt, output.nll, laplace_nle, BIC, output.noise_hyp)

def parse_all_results():
    '''
    Creates a table of some sort?
    '''
    entries = [];
    rownames = [];
    
    colnames = ['Dataset', 'NLL', 'Kernel' ]
    for rt in gen_all_results():
        print "dataset: %s kernel: %s\n" % (rt[0], rt[-1].pretty_print())
        entries.append(['%4.1f' % rt[1], rt[-1].latex_print()])
        rownames.append(rt[0])
    
    utils.latex.table('../latex/tables/kernels.tex', rownames, colnames, entries)

def gen_all_results():
    '''Look through all the files in the results directory'''
    for r,d,f in os.walk(config.RESULTS_PATH):
        for files in f:
            if files.endswith(".txt"):
                results_filename = os.path.join(r,files)
                best_tuple = parse_results( results_filename )
                yield files.split('.')[-2], best_tuple
                
def parse_results( results_filename ):
    '''
    Returns the best kernel in an experiment output file as a ScoredKernel
    '''
    result_tuples = [ScoredKernel.parse_results_string(line.strip()) for line in open(results_filename) if line.startswith("ScoredKernel")]
    best_tuple = sorted(result_tuples, key=ScoredKernel.score)[0]
    return best_tuple

def gen_all_kfold_datasets():
    '''Look through all the files in the results directory'''
    for r,d,f in os.walk("../data/kfold_data/"):
        for files in f:
            if files.endswith(".mat"):
                yield r, files.split('.')[-2]

def main():
    '''
    Currently does nothing
    '''
    data_file = sys.argv[1];
    results_filename = sys.argv[2];
    max_depth = int(sys.argv[3]);
    k = int(sys.argv[4]);
    
    print 'Datafile=%s' % data_file
    print 'results_filename=%s' % results_filename
    print 'max_depth=%d' % max_depth
    print 'k=%d' % k
    
    #experiment(data_file, results_filename, max_depth=max_depth, k=k)    
    
def run_all_kfold(local_computation = True, skip_complete=False, zip_files=False, max_jobs=500, random_walk=False):
    if (not local_computation) and (LOCATION == 'home'):
        cblparallel.start_port_forwarding()
    data_sets = list(gen_all_kfold_datasets())
    if random_walk:
        random.shuffle(data_sets)
    for r, files in data_sets:
        # Do we need to run this test?
        if not(skip_complete and (os.path.isfile(os.path.join(RESULTS_PATH, files + "_result.txt")))):
            datafile = os.path.join(r,files + ".mat")
            output_file = os.path.join(RESULTS_PATH, files + "_result.txt")
            prediction_file = os.path.join(RESULTS_PATH, files + "_predictions.mat")
            
            perform_kernel_search(datafile, output_file, max_depth=4, k=3, description = '1 per cent Frobenius cut off', verbose=True, local_computation=local_computation, zip_files=zip_files, max_jobs=max_jobs)
            
            #k_opt, nll, laplace_nle, BIC, noise_hyp = parse_results(output_file)
            #gpml.make_predictions(k_opt.gpml_kernel_expression(), k_opt.param_vector(), datafile, prediction_file, noise_hyp, iters=30)  
            make_predictions(os.path.abspath(datafile), output_file, prediction_file)      
            
            print "Done one file!!!"  
        else:
            print 'Skipping file %s' % files
    
def run_test_kfold(local_computation = True):
    
    datafile = '../data/kfold_data/r_pumadyn512_fold_3_of_10.mat'
    output_file = '../results' + '/r_pumadyn512_fold_3_of_10_result.txt'
    if (not local_computation) and (LOCATION == 'home'):
        cblparallel.start_port_forwarding()
    perform_kernel_search(datafile, output_file, max_depth=2, k=1, description = 'J-Llo test', debug=True, local_computation=local_computation)
    prediction_file = '../results' + '/r_pumadyn512_fold_3_of_10_predictions.mat'
    make_predictions(os.path.abspath(datafile), output_file, prediction_file, local_computation=True)
                                   
    #gpml.make_predictions(best_scored_kernel.k_opt.gpml_kernel_expression(), best_scored_kernel.k_opt.param_vector(), datafile, prediction_file, best_scored_kernel.noise, iters=30)
    
    
if __name__ == '__main__':
    #main()

    run_test_kfold()
    #run_all_kfold()

