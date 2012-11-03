'''
Created on Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import flexiblekernel as fk
import subprocess, os, sys, time

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
    
def call_gpml_test():
    k = fk.CompoundKernel(fk.SqExpPeriodicKernel(1, 0, 2, 2, 2))
    gpml_kernel_string = k.gpml_kernel_expression()
    gpml_param_string = k.gpml_param_expression()
    print gpml_kernel_string
    print gpml_param_string
    matlab_location = "/misc/apps/matlab/matlabR2011b/bin/matlab"
    call = [matlab_location, "-nosplash", "-nojvm", "-nodisplay", "-r", "fprintf('Hello from MATLAB\n'); exit()"];
    subprocess.call(call);  # Call in a blocking way.
    print "done"
     
if __name__ == '__main__':
    kernel_test()
    expression_test()
    base_kernel_test()
    expand_test()
    #call_gpml_test()