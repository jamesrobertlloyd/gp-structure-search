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

import numpy as np
import pylab
import scipy.io
import sys
import getopt


def load_mat(data_file):
    '''Load a Matlab file'''
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y'], np.shape(data['X'])[1]


def try_expanded_kernels(X, y, D, seed_kernels, expand=True, verbose=False):    
    '''
    expand: if false, just tries kernels that were passed in.
    TODO: separate expansion and trying
    '''

            
    g = grammar.MultiDGrammar(D)
    print 'Seed kernels :'
    for k in seed_kernels:
        print k.pretty_print()
    if expand:
        kernels = []
        for k in seed_kernels:
            kernels = kernels + grammar.expand(k, g)
        kernels = grammar.remove_duplicates(kernels)
    else:
        kernels = seed_kernels
    kernels = grammar.remove_duplicates(kernels)
    print 'Trying the following kernels :'
    for k in kernels:
        print k.pretty_print()
            
    results = []

    # Call GPML with each of the expanded kernels
    #pylab.figure()
    for k in kernels:
        #### TODO - think about initialisation
        #init_params = np.random.normal(size=k.param_vector().size)
        init_params = k.param_vector()
        kernel_hypers, nll, nlls, laplace_nle = gpml.optimize_params(k.gpml_kernel_expression(), init_params, X, y, return_all=True, verbose=verbose)
    
        if verbose:
            print "kernel_hypers =", kernel_hypers
        print
        print "nll =", nll
        print "laplace =", laplace_nle
       
        k_opt = k.family().from_param_vector(kernel_hypers)
        if verbose:
            print k_opt.gpml_kernel_expression()
        print k_opt.pretty_print()
        if verbose:
            print '[%s]' % k_opt.param_vector()
        
        # pylab.semilogx(range(1, nlls.size+1), nlls)
        
        results.append((k_opt, nll, laplace_nle))
        
        #pylab.draw()  
        
    return results


def experiment(data_file, results_filename, max_depth=2, k=2, verbose=True):
    '''Recursively search for the best kernel'''

    X, y, D = load_mat(data_file)
    
    seed_kernels = [fk.MaskKernel(D, 0, fk.SqExpKernel(0, 0))]
    
    nll_key = 1
    laplace_key = 2
    active_key = laplace_key
    
    results = []
    for r in range(max_depth):     
        new_results = try_expanded_kernels(X, y, D=D, seed_kernels=seed_kernels, verbose=verbose)
        results = results + new_results
        
        print
        results = sorted(results, key=lambda p: p[active_key], reverse=True)
        for kernel, nll, laplace in results:
            print nll, laplace, kernel.pretty_print()
            
        seed_kernels = [r[0] for r in sorted(new_results, key=lambda p: p[active_key])[0:k]]

    # Write results to a file
    results = sorted(results, key=lambda p: p[nll_key], reverse=True)
    with open(results_filename, 'w') as outfile:
        outfile.write('Experiment results for\n datafile = %s\n max_depth = %f\n k = %f\n\n' % (data_file, max_depth, k)) 
        for kernel, nll, laplace in results:
            outfile.write( 'nll=%f, laplace=%f, kernel=%s\n' % (nll, laplace, kernel.__repr__()))


if __name__ == '__main__':
    data_file = sys.argv[1];
    results_filename = sys.argv[2];
    max_depth = int(sys.argv[3]);
    k = int(sys.argv[4]);
    
    print 'Datafile=%s' % data_file
    print 'results_filename=%s' % results_filename
    print 'max_depth=%d' % max_depth
    print 'k=%d' % k
    
    experiment(data_file, results_filename, max_depth=max_depth, k=k)
