'''
Some routines to test our kernel-handling routines.

@authors: 
          James Robert Lloyd (jrl44@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
          David Duvenaud (dkd23@cam.ac.uk)
'''

import os
import scipy.io

import flexiblekernel as fk
import gpml
import grammar

def test_kernel_eval():
    '''Tests whether we can take a string and make a kernel out of it.'''
    result_string = '''ScoredKernel(k_opt=ProductKernel([ MaskKernel(ndim=4, active_dimension=1, base_kernel=PP0Kernel(lengthscale=-3.776833, output_variance=-3.365662)), MaskKernel(ndim=4, active_dimension=2, base_kernel=CubicKernel(offset=-1.149225, output_variance=-0.604651)) ]), nll=4546.591426, laplace_nle=4531.678317, bic_nle=9108.234692, noise=[-2.26189631])'''
    k = fk.repr_string_to_kernel(result_string)
    k.pretty_print()
    
    
def test_kernel_expand():
    k = fk.Carls_Mauna_kernel()
    k_expanded = grammar.expand_kernels(1, [k])
    assert len(k_expanded) > 1
    
def test_kernel_expand_multi_d():
    D = 3
    k_base = list(fk.base_kernels(3))
    k_expanded = grammar.expand_kernels(3, k_base)
    assert len(k_expanded) > len(k_base)

def test_kernel_decompose_1d():
    '''Checks that a kernel decomposes into a sum properly'''
    # = fk.Carls_Mauna_kernel()
    #kparts = break_kernel_into_summands(k)
    
    k = fk.repr_string_to_kernel('ScoredKernel(k_opt=ProductKernel([ SumKernel([ MaskKernel(ndim=1, active_dimension=0, base_kernel=SqExpKernel(lengthscale=-1.155844, output_variance=0.053888)), MaskKernel(ndim=1, active_dimension=0, base_kernel=SqExpKernel(lengthscale=1.849842, output_variance=0.682585)), MaskKernel(ndim=1, active_dimension=0, base_kernel=SqExpKernel(lengthscale=5.202112, output_variance=3.644150)), MaskKernel(ndim=1, active_dimension=0, base_kernel=LinKernel(lengthscale=0.568343)) ]), SumKernel([ MaskKernel(ndim=1, active_dimension=0, base_kernel=SqExpPeriodicKernel(lengthscale=-0.033653, period=2.486995, output_variance=0.843627)), MaskKernel(ndim=1, active_dimension=0, base_kernel=LinKernel(lengthscale=2.847578)) ]) ]), nll=553.386337, laplace_nle=597.255308, bic_nle=1156.470807, noise=[ 0.9277132])')
    