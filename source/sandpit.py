'''
Created on Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import flexiblekernel as fk
import gpml

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
    print [k.pretty_print() for k in fk.base_kernels(5)]
    print 'base_kernel_test complete'
    
def expand_test():
    k1 = fk.MaskKernel(4, 0, fk.SqExpKernel(0, 0))
    k2 = fk.MaskKernel(4, 1, fk.SqExpPeriodicKernel(1, 1, 1))
    k3 = fk.MaskKernel(4, 2, fk.SqExpKernel(3, 4))
    k4 = fk.MaskKernel(4, 3, fk.SqExpPeriodicKernel(2, 2, 2))
    f = fk.ProductKernel(operands = [k3, k4])
    e = fk.SumKernel(operands = [k1, k2, f])
    #e = fk.CompoundKernel(operator = '+', operands = [k1, k2])
    #print e.polish_expression()
    print ''
    for f in e.expand(4):
        print f.gpml_param_expression()
    print 'expand_test complete'
    

def load_mauna():
    data_file = '../data/mauna.mat'
    data = scipy.io.loadmat(data_file)
    return data['X'], data['y']

def call_gpml_test():
    k = fk.SqExpKernel(0, 0)
    print k.gpml_kernel_expression()
    print k.pretty_print()
    print '[%s]' % k.param_vector()

    X, y = load_mauna()

    kernel_hypers, nll = gpml.optimize_params(k.gpml_kernel_expression(), k.param_vector(), X, y)

    print "done"
    

if __name__ == '__main__':
    #kernel_test()
    #expression_test()
    #base_kernel_test()
    #expand_test()
    call_gpml_test()
