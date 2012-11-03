'''
Created Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import numpy as np

class SqExpKernelFamily:
    def from_param_vector(self, params):
        output_variance, lengthscale = params
        return SqExpKernel(output_variance, lengthscale)
    
    def num_params(self):
        return 2

class SqExpKernel:
    def __init__(self, lengthscale, output_variance):
        self.output_variance = output_variance
        self.lengthscale = lengthscale
        
    def family(self):
        return SqExpKernelFamily()
        
    def gpml_kernel_expression(self):
        return '@covSEiso'
    
    def english_name(self):
        return 'SqExp'
    
    def param_vector(self):
        # order of args matches GPML
        return np.array([self.lengthscale, self.output_variance])

    def copy(self):
        return SqExpKernel(self.lengthscale, self.output_variance)
    
    def __repr__(self):
        return 'SqExpKernel(lengthscale=%f, output_variance=%f)' % (self.lengthscale, self.output_variance)
    
    def pretty_print(self):
        return 'SqExp(ell=%1.1f, sf=%1.1f)' % (self.lengthscale, self.output_variance)

class SqExpPeriodicKernelFamily:
    def from_param_vector(self, params):
        output_variance, period, lengthscale = params
        return SqExpPeriodicKernel(output_variance, period, lengthscale)
    
    def num_params(self):
        return 3
    
class SqExpPeriodicKernel:
    def __init__(self, lengthscale, period, output_variance):
        self.lengthscale = lengthscale
        self.period = period
        self.output_variance = output_variance
        
    def gpml_kernel_expression(self):
        return '@covPeriodic'
    
    def english_name(self):
        return 'Periodic'
    
    def param_vector(self):
        # order of args matches GPML
        return np.array([self.lengthscale, self.period, self.output_variance])

    def copy(self):
        return SqExpKernel(self.lengthscale, self.period, self.output_variance)
    
    def __repr__(self):
        return 'SqExpPeriodicKernel(lengthscale=%f, period=%f, output_variance=%f)' % \
            (self.lengthscale, self.period, self.output_variance)
    
    def pretty_print(self):
        return 'Periodic(ell=%1.1f, p=%1.1f, sf=%1.1f)' % (self.lengthscale, self.period, self.output_variance)
    
class MaskKernelFamily:
    def __init__(self, ndim, active_dimension, base_kernel_family):
        assert 0 <= active_dimension < ndim
        self.ndim = ndim
        self.active_dimension = active_dimension    # first dimension is 0
        self.base_kernel_family = base_kernel_family
        
    def from_param_vector(self, params):
        return MaskKernel(self.ndim, self.active_dimension, self.base_kernel_family.from_param_vector(params))
    
    def num_params(self):
        return self.base_kernel_family.num_params()
    
class MaskKernel:
    def __init__(self, ndim, active_dimension, base_kernel):
        assert 0 <= active_dimension < ndim
        self.ndim = ndim
        self.active_dimension = active_dimension    # first dimension is 0
        self.base_kernel = base_kernel
        
    def family(self):
        return MaskKernelFamily(self.ndim, self.active_dimension, self.base_kernel.family())
        
    def gpml_kernel_expression(self):
        dim_vec = np.zeros(self.ndim, dtype=int)
        dim_vec[self.active_dimension] = 1
        dim_vec_str = '[' + ' '.join(map(str, dim_vec)) + ']'
        return '{@covMask, {%s, %s}}' % (dim_vec_str, self.base_kernel.gpml_kernel_expression())
    
    def pretty_print(self):
        return 'Mask(%d, %s)' % (self.active_dimension, self.base_kernel.pretty_print())
    

class SumKernelFamily:
    def __init__(self, operands):
        self.operands = operands
        
    def from_param_vector(self, params):
        start = 0
        ops = []
        for e in self.operands:
            end = start + e.num_params()
            ops.append(e.from_param_vector(params[start:end]))
        return SumKernel(ops)
    
    def num_params(self):
        return sum([e.num_params() for e in self.operands])

class SumKernel:
    def __init__(self, operands):
        self.operands = operands
        
    def family(self):
        return SumKernelFamily([e.family() for e in self.operands])
        
    def pretty_print(self):
        return '( ' + ' + '.join([e.pretty_print() for e in self.operands]) + ' ) '
    
    def gpml_kernel_expression(self):
        return '{@covSum, {%s}}' % ', '.join(e.gpml_kernel_expression() for e in self.operands)
    
    def copy(self):
        return SumKernel([e.copy() for e in self.operands])

    def param_vector(self):
        return np.concatenate([e.param_vector() for e in self.operands])
    
class ProductKernelFamily:
    def __init__(self, operands):
        self.operands = operands
        
    def from_param_vector(self, params):
        start = 0
        ops = []
        for e in self.operands:
            end = start + e.num_params()
            ops.append(e.from_param_vector(params[start:end]))
        return ProductKernel(ops)
    
    def num_params(self):
        return sum([e.num_params() for e in self.operands])
    
class ProductKernel:
    def __init__(self, operands):
        self.operands = operands
        
    def family(self):
        return ProductKernelFamily([e.family() for e in self.operands])
        
    def pretty_print(self):
        return '( ' + ' * '.join([e.pretty_print() for e in self.operands]) + ' ) '
    
    def gpml_kernel_expression(self):
        return '{@covProd, {%s}}' % ', '.join(e.gpml_kernel_expression() for e in self.operands)
    
    def copy(self):
        return ProductKernel([e.copy() for e in self.operands])

    def param_vector(self):
        return np.concatenate([e.param_vector() for e in self.operands])

            
def base_kernels(ndim=1):
    '''
    Generator of all base kernels for a certain dimensionality of data
    '''
    for dim in range(ndim):
        # Make up default arguments.
        yield MaskKernel(ndim, dim, SqExpKernel(0, 0))
        yield MaskKernel(ndim, dim, SqExpPeriodicKernel(0, 0, 0))
            
