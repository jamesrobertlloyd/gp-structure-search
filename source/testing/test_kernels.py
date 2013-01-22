'''
Some routines to test our kernel-handling routines.

@authors: 
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          David Duvenaud (dkd23@cam.ac.uk)
'''

import gpml
import flexiblekernel as fk

import os
import scipy.io

def test_kernel_eval():
    '''Tests whether we can take a string and make a kernel out of it.'''
    result_string = '''ScoredKernel(k_opt=ProductKernel([ MaskKernel(ndim=4, active_dimension=1, base_kernel=PP0Kernel(lengthscale=-3.776833, output_variance=-3.365662)), MaskKernel(ndim=4, active_dimension=2, base_kernel=CubicKernel(offset=-1.149225, output_variance=-0.604651)) ]), nll=4546.591426, laplace_nle=4531.678317, bic_nle=9108.234692, noise=[-2.26189631])'''
    k = fk.repr_string_to_kernel(result_string)
    k.pretty_print()
    
    
    
