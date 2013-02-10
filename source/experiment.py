'''
Main file for setting up experiments, and compiling results.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
Created Jan 2013          
'''

from collections import namedtuple
from itertools import izip
import numpy as np
nax = np.newaxis
import os
import random
import scipy.io

import flexiblekernel as fk
from flexiblekernel import ScoredKernel
import grammar
import gpml
import utils.latex
import cblparallel
from cblparallel.util import mkstemp_safe
from config import *
import job_controller as jc   # This might be a good, if hacky, place to switch to an MIT controller.
import utils.misc
import re

PERIOD_HEURISTIC = 10;   # How many multiples of the smallest interval between points to initialize periods to.
FROBENIUS_CUTOFF = 0.01; # How different two matrices have to be to be considered different.

def remove_duplicates(kernels, X, n_eval=250, local_computation=True, verbose=True):
    '''
    Test the top n_eval performing kernels for equivalence, in terms of their covariance matrix evaluated on training inputs
    Assumes kernels is a list of ScoredKernel objects
    '''
    # Because this is slow, we can do it locally.
    local_computation = True

    kernels = sorted(kernels, key=ScoredKernel.score, reverse=False)
    
    # Find covariance distance for top n_eval
    n_eval = min(n_eval, len(kernels))
    distance_matrix = jc.covariance_distance(kernels[:n_eval], X, local_computation=local_computation, verbose=verbose)
    
    # Remove similar kernels
    #### TODO - What is a good heuristic for determining equivalence?
    ####      - Currently using Frobenius norm - truncate if Frobenius norm is below a fraction of the average
    cut_off = distance_matrix.mean() * FROBENIUS_CUTOFF
    equivalence_graph = distance_matrix < cut_off
    # For all kernels (best first)
    for i in range(n_eval):
        # For all other worse kernels
        for j in range(i+1, n_eval):
            if equivalence_graph[i,j]:
                # Destroy the inferior duplicate
                kernels[j] = None

    kernels = [k for k in kernels if k is not None]
    kernels = sorted(kernels, key=ScoredKernel.score, reverse=True)
    return kernels
 
def remove_nan_scored_kernels(scored_kernels):    
    return [k for k in scored_kernels if not np.isnan(k.score())] 
    
def perform_kernel_search(X, y, D, experiment_data_file_name, results_filename, y_dim=1, subset=None, max_depth=2, k=2, \
                          verbose=True, description='No description', n_rand=1, sd=4, debug=False, local_computation=False, zip_files=False, max_jobs=500, \
                          use_min_period=True):
    '''Search for the best kernel, in parallel on fear or local machine.'''

    # Initialise kernels to be all base kernels along all dimensions.
    current_kernels = list(fk.base_kernels(D))
    
    # Initialise period at a multiple of the shortest distance between points, to prevent Nyquist problems.
    if use_min_period:
        min_period = np.log([PERIOD_HEURISTIC * utils.misc.min_abs_diff(X[:,i]) for i in range(X.shape[1])])
    else:
        min_period = None
    print min_period
    
    all_results = []
    results_sequence = []     # List of lists of results, indexed by level of expansion.
    
    # Perform search
    for depth in range(max_depth):
        
        if debug==True:
            current_kernels = current_kernels[0:4]
             
        # Add random restarts to kernels
        current_kernels = fk.add_random_restarts(current_kernels, n_rand, sd, min_period=min_period)
        # Score the kernels
        new_results = jc.evaluate_kernels(current_kernels, X, y, verbose=verbose, local_computation=local_computation, zip_files=zip_files, max_jobs=max_jobs)
        # Some of the scores may have failed - remove nans to prevent sorting algorithms messing up
        new_results = remove_nan_scored_kernels(new_results)
        assert(len(new_results) > 0) # FIXME - Need correct control flow if this happens 
        # Sort the new all_results
        new_results = sorted(new_results, key=ScoredKernel.score, reverse=True)
        
        print 'All new results:'
        for result in new_results:
            print result.nll, result.laplace_nle, result.bic_nle, result.k_opt.pretty_print()
            
        # Remove near duplicates from these all_results (top m all_results only for efficiency)
        new_results = remove_duplicates(new_results, X, local_computation=local_computation, verbose=verbose)

        print 'All new results after duplicate removal:'
        for result in new_results:
            print result.nll, result.laplace_nle, result.bic_nle, result.k_opt.pretty_print()

        all_results = all_results + new_results
        all_results = sorted(all_results, key=ScoredKernel.score, reverse=True)

        results_sequence.append(all_results)
        if verbose:
            print 'Printing all results'
            for result in all_results:
                print result.nll, result.laplace_nle, result.bic_nle, result.k_opt.pretty_print()
        
        # Extract the best k kernels from the new all_results
        best_kernels = [r.k_opt for r in sorted(new_results, key=ScoredKernel.score)[0:k]]
        current_kernels = grammar.expand_kernels(D, best_kernels, verbose=verbose, debug=debug)
        
        if debug==True:
            current_kernels = current_kernels[0:4]

    # Write all_results to a file.
    all_results = sorted(all_results, key=ScoredKernel.score, reverse=True)
    with open(results_filename, 'w') as outfile:
        outfile.write('Experiment all_results for\n datafile = %s\n y_dim = %d\n subset = %s\n max_depth = %f\n k = %f\n Description = %s\n\n' \
                      % (experiment_data_file_name, y_dim, subset, max_depth, k, description)) 
        for (i, all_results) in enumerate(results_sequence):
            outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
            for result in all_results:
                print >> outfile, result  



def parse_results( results_filename, max_level=None ):
    '''
    Returns the best kernel in an experiment output file as a ScoredKernel
    '''
    # Read relevant lines of file
    lines = []
    with open(results_filename) as results_file:
        for line in results_file:
            if line.startswith("ScoredKernel"):
                lines.append(line)
            elif (not max_level is None) and (len(re.findall('Level [0-9]+', line)) > 0):
                level = int(line.split(' ')[2])
                if level > max_level:
                    break
    #result_tuples = [fk.repr_string_to_kernel(line.strip()) for line in open(results_filename) if line.startswith("ScoredKernel")]
    result_tuples = [fk.repr_string_to_kernel(line.strip()) for line in lines]
    best_tuple = sorted(result_tuples, key=ScoredKernel.score)[0]
    return best_tuple

def gen_all_datasets(dir):
    """Looks through all .mat files in a directory, or just returns that file if it's only one."""
    if dir.endswith(".mat"):
        (r, f) = os.path.split(dir)
        return [(r, f.split('.')[-2])]
    
    file_list = []
    for r,d,f in os.walk(dir):
        for files in f:
            if files.endswith(".mat"):
                file_list.append((r, files.split('.')[-2]))
    file_list.sort()
    return file_list



# Define a class that keeps track of all the options for an experiment.
Experiment = namedtuple("Experiment",
                        'description,'
                        'data_dir,'
                        'max_depth, '
                        'random_order,'
                        'k,'
                        'debug, '
                        'local_computation,' 
                        'n_rand, ' 
                        'sd, '
                        'max_jobs, ' 
                        'verbose, '
                        'make_predictions, '
                        'skip_complete'
                        );

def run_experiment_file(filename):       
    expstring = open(filename, 'r').read()
    exp = eval(expstring)
    
    print "Running experiment:"
    for field, val in izip(exp._fields, exp):
        print field, '=', val, ','
    
def other():
    data_sets = list(gen_all_datasets(exp.data_dir))

    if exp.random_order:
        random.shuffle(data_sets)

    for r, file in data_sets:
        # Check if this experiment has already been done.
        if not(exp.skip_complete and (os.path.isfile(os.path.join(RESULTS_PATH, file + "_result.txt")))):
            print 'Experiment %s' % file
            data_file = os.path.join(r, file + ".mat")

            perform_experiment(data_file, exp )
            print "Finished file %s" % file
        else:
            print 'Skipping file %s' % file

    os.system('reset')  # Stop terminal from going invisible.   
    
def perform_experiment(data_file, exp):
    output_file = os.path.join(RESULTS_PATH, data_file + "_result.txt")
    
    if exp.make_predictions:        
        X, y, D, Xtest, ytest = gpml.load_mat(data_file, y_dim=1)
        prediction_file = os.path.join(RESULTS_PATH, data_file + "_predictions.mat")
    else:
        X, y, D = gpml.load_mat(data_file, y_dim=1)
        
    perform_kernel_search(X, y, D, data_file, output_file, max_depth=exp.max_depth, k=exp.k,
                          description=exp.description, debug=exp.debug, local_computation=exp.local_computation,
                          n_rand=exp.n_rand, sd=exp.sd, max_jobs=exp.max_jobs, verbose=exp.verbose)
    best_scored_kernel = parse_results(output_file)
    
    if exp.make_predictions:
        predictions = jc.make_predictions(X, y, Xtest, ytest, best_scored_kernel, local_computation=local_computation, max_jobs=max_jobs, verbose=verbose)
        scipy.io.savemat(prediction_file, predictions, appendmat=False)
        
    os.system('reset')  # Stop terminal from going invisible.
   

def run_debug_kfold(local_computation = True, max_jobs=600, verbose=True):
    """This is a quick debugging function."""
    run_experiment_file('../experiments/debug_example.py')
    
