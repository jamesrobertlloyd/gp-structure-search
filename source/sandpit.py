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
        
        
    
    

if __name__ == '__main__':
    #kernel_test()
    #expression_test()
    #base_kernel_test()
    expand_test()
    #call_gpml_test()
