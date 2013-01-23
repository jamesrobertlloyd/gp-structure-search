'''
Main file for setting up experiments, and compiling results.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
Created Jan 2013          
'''

# This might be a good place to switch to an MIT controller
from job_controller import *
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
    
#def remove_nans_from_list(a_list):
#    return [element for element in a_list if not np.isnan(element)]  
    
def remove_nan_scored_kernels(scored_kernels):    
    return [k for k in scored_kernels if not np.isnan(k.score())] 
    
def perform_kernel_search(X, y, D, experiment_data_file_name, results_filename, y_dim=1, subset=None, max_depth=2, k=2, \
                          verbose=True, description='No description', n_rand=1, sd=2, local_computation=False, debug=False, zip_files=False, max_jobs=500):
    '''Recursively search for the best kernel, in parallel on fear or local machine.'''

    # Load data - input, output and number of input dimensions
    #X, y, D = gpml.load_mat(data_file, y_dim)
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
        current_kernels = fk.add_random_restarts(current_kernels, n_rand, sd)
        # Score the kernels
        new_results = evaluate_kernels(current_kernels, X, y, verbose=verbose, local_computation=local_computation, zip_files=zip_files, max_jobs=max_jobs)
        # Some of the scores may have failed - remove nans to prevent sorting algorithms messing up
        new_results = remove_nan_scored_kernels(new_results)
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
            current_kernels = grammar.expand_kernels(4, best_kernels, verbose=verbose, debug=debug)
        else:
            current_kernels = grammar.expand_kernels(D, best_kernels, verbose=verbose, debug=debug)

    # Write results to a file.
    results = sorted(results, key=ScoredKernel.score, reverse=True)
    with open(results_filename, 'w') as outfile:
        outfile.write('Experiment results for\n datafile = %s\n y_dim = %d\n subset = %s\n max_depth = %f\n k = %f\n Description = %s\n\n' \
                      % (experiment_data_file_name, y_dim, subset, max_depth, k, description)) 
        for (i, results) in enumerate(results_sequence):
            outfile.write('\n%%%%%%%%%% Level %d %%%%%%%%%%\n\n' % i)
            for result in results:
                print >> outfile, result  


def parse_all_results(folder=RESULTS_PATH, save_file='kernels.tex'):
    '''
    Creates a list of results, then sends them to be formatted into latex.
    '''
    entries = [];
    rownames = [];
    
    colnames = ['Dataset', 'NLL', 'Kernel' ]
    for rt in gen_all_results(folder):
        print "dataset: %s kernel: %s\n" % (rt[0], rt[-1].pretty_print())
        entries.append(['%4.1f' % rt[-1].nll, rt[-1].latex_print()])
        rownames.append(rt[0])
    
    utils.latex.table(''.join(['../latex/tables/', save_file]), rownames, colnames, entries)


def gen_all_results(folder=RESULTS_PATH):
    '''Look through all the files in the results directory'''
    file_list = sorted([f for (r,d,f) in os.walk(folder)][0])
    #for r,d,f in os.walk(folder):
    for files in file_list:
        if files.endswith(".txt"):
            results_filename = os.path.join(folder,files)#r
            best_tuple = parse_results( results_filename )
            yield files.split('.')[-2], best_tuple
                

def parse_results( results_filename ):
    '''
    Returns the best kernel in an experiment output file as a ScoredKernel
    '''
    result_tuples = [fk.repr_string_to_kernel(line.strip()) for line in open(results_filename) if line.startswith("ScoredKernel")]
    best_tuple = sorted(result_tuples, key=ScoredKernel.score)[0]
    return best_tuple

def gen_all_kfold_datasets():
    '''Look through all the files in the results directory'''
    for r,d,f in os.walk("../data/kfold_data/"):
        for files in f:
            if files.endswith(".mat"):
                yield r, files.split('.')[-2]

def perform_experiment(data_file, output_file, prediction_file, max_depth=8, k=1, description='Describe me!', debug=False, local_computation=True, n_rand=1, sd=2, max_jobs=500):
    #### FIXME - D is redundant
    X, y, D, Xtest, ytest = gpml.load_mat(data_file, y_dim=1)
    perform_kernel_search(X, y, D, data_file, output_file, max_depth=max_depth, k=k, description=description, debug=debug, local_computation=local_computation, n_rand=n_rand, sd=sd, max_jobs=max_jobs)
    best_scored_kernel = parse_results(output_file)
    predictions = make_predictions(X, y, Xtest, ytest, best_scored_kernel, local_computation=local_computation, max_jobs=max_jobs)
    scipy.io.savemat(prediction_file, predictions, appendmat=False)

def run_all_kfold(local_computation = True, skip_complete=False, zip_files=False, max_jobs=500, random_walk=False):
    data_sets = list(gen_all_kfold_datasets())
	#### FIXME - Comment / or make more elegant
    if random_walk:
        random.shuffle(data_sets)
    for r, files in data_sets:
        # Do we need to run this test?
        if not(skip_complete and (os.path.isfile(os.path.join(RESULTS_PATH, files + "_result.txt")))):
            data_file = os.path.join(r,files + ".mat")
            output_file = os.path.join(RESULTS_PATH, files + "_result.txt")
            prediction_file = os.path.join(RESULTS_PATH, files + "_predictions.mat")
            
            perform_experiment(data_file, output_file, prediction_file, max_depth=8, k=1, description='1 % Frobenius cut off', debug=False, local_computation=False, n_rand=1, sd=2, max_jobs=max_jobs)
            
            print "Done one file!!!"  
        else:
            print 'Skipping file %s' % files
    
  
def run_test_kfold(local_computation = True, max_jobs=600):
    #### TODO - Add description
    data_file = '../data/kfold_data/r_pumadyn512_fold_3_of_10.mat'
    output_file = '../test_results' + '/r_pumadyn512_fold_3_of_10_result.txt'
    prediction_file = '../test_results' + '/r_pumadyn512_fold_3_of_10_predictions.mat'
    #### TODO - make this always happen in cblparallel.__init__
    if (not local_computation) and (LOCATION == 'home'):
        cblparallel.start_port_forwarding()
    perform_experiment(data_file, output_file, prediction_file, max_depth=1, k=1, description='DaDu test', debug=True, local_computation=local_computation, max_jobs=max_jobs)


