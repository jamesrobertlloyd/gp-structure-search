'''
Created on Nov 2012

@author: James Robert Lloyd
'''

import flexiblekernel
import subprocess, os, sys, time

def kernel_test():
    k = flexiblekernel.BaseKernel(active_dimensions=[0,0,1,1,1])
    print k.gpml_kernel_expression()
    print k.polish_expression()
    print '[%s]' % k.gpml_param_expression()
    print 'kernel_test complete'
    
def expression_test():
    k1 = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(active_dimensions=[1,0,0,0]))
    k2 = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[0,1,0,0], params=[1,1,1]))
    k3 = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(active_dimensions=[0,0,1,0], params=[3,4]))
    k4 = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[0,0,0,1], params=[2,2,2]))
    f = flexiblekernel.CompoundKernel(operator = 'x', operands = [k3, k4])
    e = flexiblekernel.CompoundKernel(operator = '+', operands = [k1, k2, f])
    print e.polish_expression()
    print e.gpml_kernel_expression()
    print '[%s]' % e.gpml_param_expression()
    print 'expression_test complete'

def base_kernel_test():
    print [k.polish_expression() for k in flexiblekernel.base_kernels(5)]
    print 'base_kernel_test complete'
    
def expand_test():
    k1 = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(active_dimensions=[1,0,0,0]))
    k2 = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[0,1,0,0], params=[1,1,1]))
    k3 = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(active_dimensions=[0,0,1,0], params=[3,4]))
    k4 = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[0,0,0,1], params=[2,2,2]))
    f = flexiblekernel.CompoundKernel(operator = 'x', operands = [k3, k4])
    e = flexiblekernel.CompoundKernel(operator = '+', operands = [k1, k2, f])
    #e = flexiblekernel.CompoundKernel(operator = '+', operands = [k1, k2])
    print e.polish_expression()
    print ''
    for f in e.expand(4):
        print f.gpml_param_expression()
    print 'expand_test complete'
    
def call_gpml_test():
    k = flexiblekernel.CompoundKernel(flexiblekernel.BaseKernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[1], params=[2,2,2]))
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