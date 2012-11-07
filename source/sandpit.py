'''
Created on Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import flexiblekernel as fk
import grammar
import gpml

import numpy as np
import pylab
import scipy.io
import sys

def kernel_test():
    k = fk.MaskKernel(4, 3, fk.SqExpKernel(0, 0))
    print k.gpml_kernel_expression()
    print k.pretty_print()
    print '[%s]' % k.param_vector()
    print 'kernel_test complete'
    
def expression_test():
    k1 = fk.MaskKernel(4, 0, fk.SqExpKernel(0, 0))
    k2 = fk.MaskKernel(4, 1, fk.SqExpPeriodicKernel(1, 1, 1))
    k3 = fk.MaskKernel(4, 2, fk.SqExpKernel(3, 4))
    k4 = fk.MaskKernel(4, 3, fk.SqExpPeriodicKernel(2, 2, 2))
    f = fk.ProductKernel(operands = [k3, k4])
    e = fk.SumKernel(operands = [k1, k2, f])
    print e.pretty_print()
    print e.gpml_kernel_expression()
    print '[%s]' % e.param_vector()
    print 'expression_test complete'

def base_kernel_test():
    print [k.pretty_print() for k in fk.base_kernels(1)]
    print 'base_kernel_test complete'
    
def expand_test():  
    k1 = fk.SqExpKernel(1, 1)
    k2 = fk.SqExpPeriodicKernel(2, 2, 2)
    e = fk.SumKernel([k1, k2])
    
    g = grammar.OneDGrammar()
    
    print ''
    for f in grammar.expand(e, g):
        #print f
        print f.pretty_print()
        print grammar.canonical(f).pretty_print()
        print
        
    print '   ***** duplicates removed  *****'
    print
    
    kernels = grammar.expand(e, g)
    for f in grammar.remove_duplicates(kernels):
        print f.pretty_print()
        print
        
    print '%d originally, %d without duplicates' % (len(kernels), len(grammar.remove_duplicates(kernels)))
        
    print 'expand_test complete'
    
def expand_test2():    
    k1 = fk.MaskKernel(2, 0, fk.SqExpKernel(1, 1))
    k2 = fk.MaskKernel(2, 1, fk.SqExpPeriodicKernel(2, 2, 2))
    e = fk.SumKernel([k1, k2])
    
    g = grammar.MultiDGrammar(2)
    
    print ''
    for f in grammar.expand(e, g):
        print f.pretty_print()
        print grammar.canonical(f).pretty_print()
        print
        
    print '   ***** duplicates removed  *****'
    print
    
    kernels = grammar.expand(e, g)
    for f in grammar.remove_duplicates(kernels):
        print f.pretty_print()
        print
        
    print '%d originally, %d without duplicates' % (len(kernels), len(grammar.remove_duplicates(kernels)))
        
    print 'expand_test complete'
    

def load_mauna():
    '''2011 Mauna dataset.'''
    
    data_file = '../data/mauna.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']

def load_mauna_original():
    '''
    Original Mauna dataset made to match the experiments from Carl's book.
    For details, see data/preprocess_mauna_2004.m
    '''
    
    data_file = '../data/mauna2003.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']


def call_gpml_test():
    
    np.random.seed(0)
    
    k = fk.SumKernel([fk.SqExpKernel(0, 0), fk.SqExpKernel(0, 0)])
    print k.gpml_kernel_expression()
    print k.pretty_print()
    print '[%s]' % k.param_vector()

    X, y = load_mauna()
    
    N_orig = X.shape[0]
    X = X[:N_orig//3, :]
    y = y[:N_orig//3, :]

    results = []

    pylab.figure()
    for i in range(15):
        init_params = np.random.normal(size=k.param_vector().size)
        #kernel_hypers, nll, nlls = gpml.optimize_params(k.gpml_kernel_expression(), k.param_vector(), X, y, return_all=True)
        kernel_hypers, nll, nlls = gpml.optimize_params(k.gpml_kernel_expression(), init_params, X, y, return_all=True)
    
        print "kernel_hypers =", kernel_hypers
        print "nll =", nll
        
        k_opt = k.family().from_param_vector(kernel_hypers)
        print k_opt.gpml_kernel_expression()
        print k_opt.pretty_print()
        print '[%s]' % k_opt.param_vector()
        
        pylab.semilogx(range(1, nlls.size+1), nlls)
        
        results.append((kernel_hypers, nll))
        
        pylab.draw()
    
    print
    print
    results = sorted(results, key=lambda p: p[1])
    for kernel_hypers, nll in results:
        print nll, kernel_hypers
        
    print "done"
        



def sample_mauna_best():
    # This kernel was chosen from a run of Mauna datapoints.
    kernel = ( fk.SqExpKernel(-0.7, -1.3) + fk.SqExpKernel(4.8, 2.3) ) * \
             ( fk.SqExpKernel(3.0, 0.5) + fk.SqExpPeriodicKernel(0.4, -0.0, -0.9) ) 
        
    X = np.linspace(0,50,500)
    
    # Todo: set random seed.
    sample = gpml.sample_from_gp_prior(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sample)
    pylab.title('( SqExp(ell=-0.7, sf=-1.3) + SqExp(ell=4.8, sf=2.3) ) \n x ( SqExp(ell=3.0, sf=0.5) + Periodic(ell=0.4, p=-0.0, sf=-0.9) )')
    
    

    
    
def sample_Carls_kernel():
    kernel = fk.Carls_Mauna_kernel()
        
    X = np.linspace(0,50,500)
    
    # Todo: set random seed.
    sample = gpml.sample_from_gp_prior(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sample)    
    pylab.title('Carl''s kernel');


def compare_kernels_experiment():
    kernel1 = fk.Carls_Mauna_kernel()
    kernel2  = ( fk.SqExpKernel(-0.7, -1.3) + fk.SqExpKernel(4.8, 2.3) ) * \
               ( fk.SqExpKernel(3.0, 0.5) + fk.SqExpPeriodicKernel(0.4, -0.0, -0.9) ) 
             
    X, y = load_mauna_original()
     
    print "Carl's kernel"
    print kernel1.pretty_print()
    kernel_hypers1, nll1 = gpml.optimize_params(kernel1.gpml_kernel_expression(), kernel1.param_vector(), \
                                                X, y, noise=np.log(0.19), iters=100 )
    k1_opt = kernel1.family().from_param_vector(kernel_hypers1)
    print k1_opt.pretty_print()   
    print "Carl's NLL =", nll1 
    
    print "Our kernel"
    print kernel1.pretty_print()
    kernel_hypers2, nll2 = gpml.optimize_params(kernel2.gpml_kernel_expression(), kernel2.param_vector(), \
                                                X, y, noise=np.log(0.19), iters=100)
    k2_opt = kernel2.family().from_param_vector(kernel_hypers2)
    print k2_opt.pretty_print()            

    print "Our NLL =", nll2
    
         

def try_expanded_kernels(X, y, D, seed_kernels, verbose=False):    
    g = grammar.MultiDGrammar(D)
    print 'Seed kernels :'
    for k in seed_kernels:
        print k.pretty_print()
    kernels = []
    for k in seed_kernels:
        kernels = kernels + grammar.expand(k, g)
    kernels = grammar.remove_duplicates(kernels)
    print 'Trying the following kernels :'
    for k in kernels:
        print k.pretty_print()
            
    results = []

    # Call GPML with each of the expanded kernels
    pylab.figure()
    for k in kernels:
        #### TODO - think about initialisation
        #init_params = np.random.normal(size=k.param_vector().size)
        init_params = k.param_vector()
        kernel_hypers, nll, nlls = gpml.optimize_params(k.gpml_kernel_expression(), init_params, X, y, return_all=True, verbose=verbose)
    
        if verbose:
            print "kernel_hypers =", kernel_hypers
        print
        print "nll =", nll
        
        BIC = 2 * nll + len(k.param_vector()) * np.log(len(y))
        print "BIC =", BIC
        
        k_opt = k.family().from_param_vector(kernel_hypers)
        if verbose:
            print k_opt.gpml_kernel_expression()
        print k_opt.pretty_print()
        if verbose:
            print '[%s]' % k_opt.param_vector()
        
        pylab.semilogx(range(1, nlls.size+1), nlls)
        
        results.append((k_opt, nll, BIC))
        
        pylab.draw()  
        
    return results

def simple_mauna_experiment():
    '''A first version of an experiment learning kernels'''
    
    seed_kernels = [fk.SqExpKernel(0, 0)]
    
    X, y = load_mauna_original()
    #N_orig = X.shape[0]  # subsample data.
    #X = X[:N_orig//3, :]
    #y = y[:N_orig//3, :] 
    
    max_depth = 4
    k = 4    # Expand k best
    nll_key = 1
    BIC_key = 2
    
    
    results = []
    for dummy in range(max_depth):     
        new_results = try_expanded_kernels(X, y, D=2, seed_kernels=seed_kernels, verbose=False)
        results = results + new_results
        
        print
        results = sorted(results, key=lambda p: p[nll_key], reverse=True)
        for kernel, nll, BIC in results:
            print nll, BIC, kernel.pretty_print()
            
        seed_kernels = [r[0] for r in sorted(new_results, key=lambda p: p[nll_key])[0:k]]

    
def plot_Carls_kernel():
    kernel = fk.Carls_Mauna_kernel()
        
    X = np.linspace(0,10,1000)
    sigma = gpml.plot_kernel(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sigma)    
    pylab.title('Carl''s kernel');
    
    
def plot_our_kernel():
    kernel  = ( fk.SqExpKernel(-0.7, -1.3) + fk.SqExpKernel(4.8, 2.3) ) * \
              ( fk.SqExpKernel(3.0, 0.5) + fk.SqExpPeriodicKernel(0.4, -0.0, -0.9) ) 
        
    X = np.linspace(0,10,1000)
    sigma = gpml.plot_kernel(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sigma)    
    pylab.title('Our kernel');  
    
def load_simple_gef_load():
    '''Zone 1 and temperature station 2'''
    
    data_file = '../data/gef_load_simple.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']  
    
def simple_gef_load_experiment(verbose=True):
    '''A first version of an experiment learning kernels'''
    
    seed_kernels = [fk.MaskKernel(2, 0, fk.SqExpKernel(0, 0)),
                    fk.MaskKernel(2, 1, fk.SqExpKernel(0, 0))]
    
    X, y = load_simple_gef_load()
    # subsample data.
    X = X[0:99, :]
    y = y[0:99, :] 
    
    max_depth = 5
    k = 2    # Expand k best
    nll_key = 1
    BIC_key = 2
    active_key = BIC_key
    
    
    results = []
    for dummy in range(max_depth):     
        new_results = try_expanded_kernels(X, y, D=2, seed_kernels=seed_kernels, verbose=verbose)
        results = results + new_results
        
        print
        results = sorted(results, key=lambda p: p[active_key], reverse=True)
        for kernel, nll, BIC in results:
            print nll, BIC, kernel.pretty_print()
            
        seed_kernels = [r[0] for r in sorted(new_results, key=lambda p: p[active_key])[0:k]]

if __name__ == '__main__':
    #kernel_test()
    #expression_test()
    #base_kernel_test()
    #expand_test()
    #call_gpml_test()
    #sample_from_gp_prior()
    if sys.flags.debug or __debug__:
        print 'Debug mode'
