'''
Main file for setting up experiments, and compiling results.

@authors: David Duvenaud (dkd23@cam.ac.uk)
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          
Created Jan 2013          
'''

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
from job_controller import *   # This might be a good, if hacky, place to switch to an MIT controller.
import utils.misc

       
def remove_duplicates(kernels, X, n_eval=250, local_computation=True):
    '''
    Test the top n_eval performing kernels for equivalence, in terms of their covariance matrix evaluated on training inputs
    Assumes kernels is a list of ScoredKernel objects
    '''
    #### HACK - this needs lots of computing power - do it locally with multi-threading
    local_computation = True
    # Sort
    kernels = sorted(kernels, key=ScoredKernel.score, reverse=False)
    # Find covariance distance for top n_eval
    n_eval = min(n_eval, len(kernels))
    distance_matrix = covariance_distance(kernels[:n_eval], X, local_computation=local_computation)
    # Remove similar kernels
    #### TODO - What is a good heuristic for determining equivalence?
    ####      - Currently using Frobenius norm - truncate if Frobenius norm is below 1% of average
    cut_off = distance_matrix.mean() / 100.0
    equivalence_graph = distance_matrix < cut_off
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
 
def remove_nan_scored_kernels(scored_kernels):    
    return [k for k in scored_kernels if not np.isnan(k.score())] 
    
def perform_kernel_search(X, y, D, experiment_data_file_name, results_filename, y_dim=1, subset=None, max_depth=2, k=2, \
                          verbose=True, description='No description', n_rand=1, sd=2, local_computation=False, debug=False, zip_files=False, max_jobs=500, \
                          use_min_period=True):
    '''Recursively search for the best kernel, in parallel on fear or local machine.'''

    # Load data - input, output and number of input dimensions
    #X, y, D = gpml.load_mat(data_file, y_dim)
    # Initialise kernels to be base kernels
    if debug:
        current_kernels = list(fk.test_kernels(4))
    else:
        current_kernels = list(fk.base_kernels(D))
    # Initialise minimum period to prevent Nyquist problems
    if use_min_period:
        #### FIXME - Magic numbers!
        min_period = np.log([10 * utils.misc.min_abs_diff(X[:,i]) for i in range(X.shape[1])])
    else:
        min_period = None
    print min_period
    # Initialise list of results
    results = []              # All results.
    results_sequence = []     # Results sets indexed by level of expansion.
    # Perform search
    for depth in range(max_depth):   
        # Add random restarts to kernels
        current_kernels = fk.add_random_restarts(current_kernels, n_rand, sd, min_period=min_period)
        # Score the kernels
        new_results = evaluate_kernels(current_kernels, X, y, verbose=verbose, local_computation=local_computation, zip_files=zip_files, max_jobs=max_jobs)
        # Some of the scores may have failed - remove nans to prevent sorting algorithms messing up
        new_results = remove_nan_scored_kernels(new_results)
        assert(len(new_results) > 0) # FIXME - Need correct control flow if this happens 
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


def parse_all_results(folder=D1_RESULTS_PATH, save_file='kernels.tex', one_d=False):
    '''
    Creates a list of results, then sends them to be formatted into latex.
    '''
    entries = [];
    rownames = [];
    
    colnames = ['Dataset', 'NLL', 'Kernel' ]
    for rt in gen_all_results(folder):
        print "dataset: %s kernel: %s\n" % (rt[0], rt[-1].pretty_print())
        if not one_d:
            entries.append([' %4.1f' % rt[-1].nll, ' $ %s $ ' % rt[-1].latex_print()])
        else:
            # Remove any underscored dimensions
            entries.append([' %4.1f' % rt[-1].nll, ' $ %s $ ' % re.sub('_{[0-9]+}', '', rt[-1].latex_print())])
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
                

def parse_results( results_filename, max_level=None ):
    '''
    Returns the best kernel in an experiment output file as a ScoredKernel
    '''
    #### FIXME - 'tuple' is a very uninformative name!
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

def gen_all_kfold_datasets():
    '''Look through all the files in the k fold data directory'''
    for r,d,f in os.walk("../data/kfold_data/"):
        for files in f:
            if files.endswith(".mat"):
                yield r, files.split('.')[-2]

def gen_all_1d_datasets():
    '''Look through all the files in the 1d data directory'''
    file_list = []
    for r,d,f in os.walk("../data/1d_data/"):
        for files in f:
            if files.endswith(".mat"):
                file_list.append((r, files.split('.')[-2]))
    file_list.sort()
    return file_list

def perform_experiment(data_file, output_file, prediction_file, max_depth=8, k=1, description='Describe me!', debug=False, local_computation=True, n_rand=1, sd=2, max_jobs=500):
    #### FIXME - D is redundant
    X, y, D, Xtest, ytest = gpml.load_mat(data_file, y_dim=1)
    perform_kernel_search(X, y, D, data_file, output_file, max_depth=max_depth, k=k, description=description, debug=debug, local_computation=local_computation, n_rand=n_rand, sd=sd, max_jobs=max_jobs)
    best_scored_kernel = parse_results(output_file)
    predictions = make_predictions(X, y, Xtest, ytest, best_scored_kernel, local_computation=local_computation, max_jobs=max_jobs)
    scipy.io.savemat(prediction_file, predictions, appendmat=False)
   
#### WARNING - Code duplication 
def perform_experiment_no_test_1d(data_file, output_file, max_depth=8, k=1, description='Describe me!', debug=False, local_computation=True, n_rand=1, sd=2, max_jobs=500):
    X, y, D = gpml.load_mat(data_file, y_dim=1)
    assert(D==1)
    perform_kernel_search(X, y, 1, data_file, output_file, max_depth=max_depth, k=k, description=description, debug=debug, local_computation=local_computation, n_rand=n_rand, sd=sd, max_jobs=max_jobs)
    best_scored_kernel = parse_results(output_file)

def run_all_kfold(local_computation = True, skip_complete=False, zip_files=False, max_jobs=500, random_walk=False):
    data_sets = list(gen_all_kfold_datasets())
	#### FIXME - Comment / or make more elegant
    if random_walk:
        random.shuffle(data_sets)
    else:
        data_sets.sort()
    for r, files in data_sets:
        # Do we need to run this test?
        if not(skip_complete and (os.path.isfile(os.path.join(RESULTS_PATH, files + "_result.txt")))):
            print 'Experiment %s' % files
            data_file = os.path.join(r,files + ".mat")
            output_file = os.path.join(RESULTS_PATH, files + "_result.txt")
            prediction_file = os.path.join(RESULTS_PATH, files + "_predictions.mat")
            
            perform_experiment(data_file, output_file, prediction_file, max_depth=12, k=1, description='SE, RQ, LN', debug=False, local_computation=local_computation, n_rand=1, sd=2, max_jobs=max_jobs)
            
            print "Done one file!!!"  
        else:
            print 'Skipping file %s' % files
            
def run_all_1d(local_computation=False, skip_complete=True, zip_files=False, max_jobs=500, random_walk=False, max_depth=10, k=1, sd=2, n_rand=9):
    data_sets = list(gen_all_1d_datasets())
	#### FIXME - Comment / or make more elegant
    if random_walk:
        random.shuffle(data_sets)
    else:
        data_sets.sort()
    for r, files in data_sets:
        # Do we need to run this test?
        if not(skip_complete and (os.path.isfile(os.path.join(D1_RESULTS_PATH, files + "_result.txt")))):
            print 'Experiment %s' % files
            data_file = os.path.join(r,files + ".mat")
            output_file = os.path.join(D1_RESULTS_PATH, files + "_result.txt")
            
            perform_experiment_no_test_1d(data_file, output_file, max_depth=max_depth, k=k, description='SE, PE, RQ, LN n_rand=9', debug=False, local_computation=local_computation, n_rand=n_rand, sd=sd, max_jobs=max_jobs)
            
            print "Done one file!!!"  
        else:
            print 'Skipping file %s' % files
    
  
def run_test_kfold(local_computation = True, max_jobs=600):
    #### TODO - Add description
    data_file = '../data/kfold_data/r_pumadyn512_fold_3_of_10.mat'
    output_file = '../test_results' + '/r_pumadyn512_fold_3_of_10_result.txt'
    prediction_file = '../test_results' + '/r_pumadyn512_fold_3_of_10_predictions.mat'
    perform_experiment(data_file, output_file, prediction_file, max_depth=1, k=1, description='DaDu test', debug=True, local_computation=local_computation, max_jobs=max_jobs)

def make_figures():
    X, y, D = gpml.load_mat('../data/mauna2003.mat')
    k = fk.Carls_Mauna_kernel()
    gpml.plot_decomposition(k, X, y, '../figures/decomposition/mauna_test', noise=-100000.0)

def make_all_1d_figures(folder=D1_RESULTS_PATH, max_level=None):
    data_sets = list(gen_all_1d_datasets())
    for r, file in data_sets:
        results_file = os.path.join(folder, file + "_result.txt")
        # Is the experiment complete
        if os.path.isfile(results_file):
            # Find best kernel and produce plots
            X, y, D = gpml.load_mat(os.path.join(r,file + ".mat"))
            best_kernel = parse_results(os.path.join(folder, file + "_result.txt"), max_level=max_level)
            stripped_kernel = fk.strip_masks(best_kernel.k_opt)
            if not max_level is None:
                fig_folder = os.path.join('../figures/decomposition/', (file + '_max_level_%d' % max_level))
            else:
                fig_folder = os.path.join('../figures/decomposition/', file)
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            gpml.plot_decomposition(stripped_kernel, X, y, os.path.join(fig_folder, file), noise=best_kernel.noise)
            
def make_all_1d_figures_all_depths(folder=D1_RESULTS_PATH, max_depth=8):
    make_all_1d_figures(folder=folder)
    for level in range(max_depth+1):
        make_all_1d_figures(folder=folder, max_level=level)
        
def make_kernel_description_table():
    '''A helper to generate a latex table listing all the kernels used, and their descriptions.'''
    entries = [];
    rownames = [];
    
    colnames = ['', 'Description', 'Parameters' ]
    for k in fk.base_kernel_families():
        # print "dataset: %s kernel: %s\n" % (rt[0], rt[-1].pretty_print())
        rownames.append( k.latex_print() )
        entries.append([ k.family().description(), k.family().params_description()])
    
    utils.latex.table('../latex/tables/kernel_descriptions.tex', rownames, colnames, entries, 'kernel_descriptions')
