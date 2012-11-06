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
    #k1 = fk.MaskKernel(4, 0, fk.SqExpKernel(0, 0))
    #k2 = fk.MaskKernel(4, 1, fk.SqExpPeriodicKernel(1, 1, 1))
    #k3 = fk.MaskKernel(4, 2, fk.SqExpKernel(3, 4))
    #k4 = fk.MaskKernel(4, 3, fk.SqExpPeriodicKernel(2, 2, 2))
    #f = fk.ProductKernel(operands = [k3, k4])
    #e = fk.SumKernel(operands = [k1, k2, f])
    #e = fk.CompoundKernel(operator = '+', operands = [k1, k2])
    #print e.polish_expression()
    
    #k1 = fk.SqExpKernel(0, 0)
    #k2 = fk.SqExpPeriodicKernel(1, 1, 1)
    #k3 = fk.SqExpKernel(3, 4)
    #k4 = fk.SqExpPeriodicKernel(2, 2, 2)
    #f = fk.ProductKernel(operands=[k3, k4])
    #e = fk.SumKernel(operands=[k1, k2, f])
    
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
    

def load_mauna():
    data_file = '../data/mauna.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']

def call_gpml_test():
    
    np.random.seed(0)
    
    #k = fk.SumKernel([fk.SqExpKernel(0, 0), fk.SqExpPeriodicKernel( 0, 0, 0)])
    #k = fk.SqExpKernel(0, 0)
    k = fk.SumKernel([fk.SqExpKernel(0, 0), fk.SqExpKernel(0, 0)])
    print k.gpml_kernel_expression()
    print k.pretty_print()
    print '[%s]' % k.param_vector()

    X, y = load_mauna()
    
    #X *= np.exp(1)
    #y *= np.exp(1)
    
    #X = X[::2, :]
    #y = y[::2, :]
    
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
    
    
def Carls_Mauna_kernel():
    '''
    This kernel was written about in Carl's book.
    
    The reported nll = 108.5
    '''
    theta_1 = np.log(66)  # ppm, sf of SE1 = magnitude of long term trend
    theta_2 = np.log(67)  # years, ell of SE1 = lengthscale of long term trend
    theta_6 = np.log(0.66)  # ppm, sf of RQ = magnitude of med term trend
    theta_7 = np.log(1.2)  # years, ell of RQ = lengthscale of med term trend
    theta_8 = np.log(0.78) # alpha of RQ
    theta_3 = np.log(2.4) # ppm, sf of periodic * SE
    theta_4 = np.log(90) # years, lengthscale of SE of periodic*SE
    theta_5 = np.log(1.3) # smoothness of periodic
    theta_9 = np.log(0.18) # ppm, amplitude of SE_noise
    theta_10 = np.log(1.6) # months, lengthscale of SE_noise
    theta_11 = np.log(0.19) # ppm, amplitude of independent noise
    
    kernel = fk.SqExpKernel(output_variance=theta_1, lengthscale=theta_2) \
           + fk.SqExpKernel(output_variance=theta_3, lengthscale=theta_4) * fk.SqExpPeriodicKernel(output_variance=0, period=0, lengthscale=theta_5) \
           + fk.RQKernel(lengthscale=theta_7, output_variance=theta_6, alpha=theta_8) \
           + fk.SqExpKernel(output_variance=theta_9, lengthscale=theta_10)
    
    return kernel
    
    
def sample_Carls_kernel():
    kernel = Carls_Mauna_kernel()
        
    X = np.linspace(0,50,500)
    
    # Todo: set random seed.
    sample = gpml.sample_from_gp_prior(kernel, X)
    
    pylab.figure()
    pylab.plot(X, sample)    
    pylab.title('Carl''s kernel');


        
EVAL_LIKELIHOODS_CODE = r"""
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
hyp.lik = log(var(y)/10)

nll = gp(hyp, @infExact, meanfunc, covfunc, likfunc, X, y);

save( '%(writefile)s', 'nll' );
exit();
"""        

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
    
    X, y = load_mauna()
    N_orig = X.shape[0]  # subsample data.
    X = X[:N_orig//3, :]
    y = y[:N_orig//3, :] 
    
    max_depth = 4
    k = 2    # Expand k best
    nll_key = 1
    BIC_key = 2
    
    
    results = []
    for dummy in range(max_depth):     
        new_results = try_expanded_kernels(X, y, D=2, seed_kernels=seed_kernels, verbose=False)
        results = results + new_results
        
        print
        results = sorted(results, key=lambda p: p[BIC_key], reverse=True)
        for kernel, nll, BIC in results:
            print nll, BIC, kernel.pretty_print()
            
        seed_kernels = [r[0] for r in sorted(new_results, key=lambda p: p[BIC_key])[0:k]]
    

if __name__ == '__main__':
    #kernel_test()
    #expression_test()
    #base_kernel_test()
    #expand_test()
    #call_gpml_test()
    #sample_from_gp_prior()
    if sys.flags.debug or __debug__:
        print 'Debug mode'
