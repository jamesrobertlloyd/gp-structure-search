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
    
def test_change_kernel_eval():
    '''Tests whether we can take a string of a changepoint kernel.'''
    result_string = '''ChangeKernel(steepness=2.0, location=-1.2)'''
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
    sk = fk.repr_string_to_kernel('ScoredKernel(k_opt=ProductKernel([ MaskKernel(ndim=1, active_dimension=0, base_kernel=RQKernel(lengthscale=4.853529, output_variance=-0.648382, alpha=0.457387)), SumKernel([ ProductKernel([ MaskKernel(ndim=1, active_dimension=0, base_kernel=SqExpPeriodicKernel(lengthscale=1.395371, period=-3.990523, output_variance=0.565365)), MaskKernel(ndim=1, active_dimension=0, base_kernel=LinKernel(offset=0.000420, lengthscale=-0.120045)) ]), ProductKernel([ MaskKernel(ndim=1, active_dimension=0, base_kernel=LinKernel(offset=0.802417, lengthscale=3.350816)), SumKernel([ MaskKernel(ndim=1, active_dimension=0, base_kernel=SqExpPeriodicKernel(lengthscale=-3.899540, period=0.087011, output_variance=2.430187)), MaskKernel(ndim=1, active_dimension=0, base_kernel=RQKernel(lengthscale=3.865315, output_variance=4.028649, alpha=-5.060996)), ProductKernel([ MaskKernel(ndim=1, active_dimension=0, base_kernel=SqExpPeriodicKernel(lengthscale=3.251723, period=1.540000, output_variance=-2.497487)), MaskKernel(ndim=1, active_dimension=0, base_kernel=LinKernel(offset=-1.424416, lengthscale=-1.732677)) ]) ]) ]) ]) ]), nll=558.339977, laplace_nle=-266.580399, bic_nle=1216.076221, noise=[ 1.66059002])')    
    k = fk.strip_masks(sk.k_opt)
    correct_answer = ['RQ \\times Per \\times Lin', 'RQ \\times Lin \\times Per', 'RQ \\times Lin \\times RQ', 'RQ \\times Lin \\times Per \\times Lin']
    kd = fk.break_kernel_into_summands(k)
    assert( [k.latex_print() for k in kd] == correct_answer )
