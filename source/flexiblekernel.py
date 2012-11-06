'''
Created Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import numpy as np
import termcolor

import config

PAREN_COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

def paren_colors():
    if config.COLOR_SCHEME == 'dark':
        return ['red', 'green', 'cyan', 'magenta', 'yellow']
    elif config.COLOR_SCHEME == 'light':
        return ['red', 'green', 'blue', 'cyan', 'magenta']
    else:
        raise RuntimeError('Unknown color scheme: %s' % config.COLOR_SCHEME)

def colored(text, depth):
    colors = paren_colors()
    color = colors[depth % len(colors)]
    return termcolor.colored(text, color, attrs=['bold'])

class KernelFamily:
    pass

class Kernel:
    pass

class BaseKernelFamily(KernelFamily):
    pass

class BaseKernel(Kernel):#
    pass

class SqExpKernelFamily(BaseKernelFamily):
    def from_param_vector(self, params):
        output_variance, lengthscale = params
        return SqExpKernel(output_variance, lengthscale)
    
    def num_params(self):
        return 2
    
    def pretty_print(self):
        return colored('SqExp', self.depth())
    
    def default(self):
        return SqExpKernel(0., 0.)
    
    def __cmp__(self, other):
        assert isinstance(other, KernelFamily)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return 0
    
    def depth(self):
        return 0

class SqExpKernel(BaseKernel):
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
        return colored('SqExp(ell=%1.1f, sf=%1.1f)' % (self.lengthscale, self.output_variance), self.depth())
    
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp((self.lengthscale, self.output_variance), (other.lengthscale, other.output_variance))
    
    def depth(self):
        return 0


class SqExpPeriodicKernelFamily(BaseKernelFamily):
    def from_param_vector(self, params):
        output_variance, period, lengthscale = params
        return SqExpPeriodicKernel(output_variance, period, lengthscale)
    
    def num_params(self):
        return 3
    
    def pretty_print(self):
        return colored('Periodic', self.depth())
    
    def default(self):
        return SqExpPeriodicKernel(0., 0., 0.)
    
    def __cmp__(self, other):
        assert isinstance(other, KernelFamily)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return 0
    
    def depth(self):
        return 0
    
    
class SqExpPeriodicKernel(BaseKernel):
    def __init__(self, lengthscale, period, output_variance):
        self.lengthscale = lengthscale
        self.period = period
        self.output_variance = output_variance
        
    def family(self):
        return SqExpPeriodicKernelFamily()
        
    def gpml_kernel_expression(self):
        return '@covPeriodic'
    
    def english_name(self):
        return 'Periodic'
    
    def param_vector(self):
        # order of args matches GPML
        return np.array([self.lengthscale, self.period, self.output_variance])

    def copy(self):
        return SqExpPeriodicKernel(self.lengthscale, self.period, self.output_variance)
    
    def __repr__(self):
        return 'SqExpPeriodicKernel(lengthscale=%f, period=%f, output_variance=%f)' % \
            (self.lengthscale, self.period, self.output_variance)
    
    def pretty_print(self):
        return colored('Periodic(ell=%1.1f, p=%1.1f, sf=%1.1f)' % (self.lengthscale, self.period, self.output_variance),
                       self.depth())
    
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp((self.lengthscale, self.period, self.output_variance), 
                   (other.lengthscale, other.period, other.output_variance))
        
    def depth(self):
        return 0
    
    
class MaskKernelFamily(KernelFamily):
    def __init__(self, ndim, active_dimension, base_kernel_family):
        assert 0 <= active_dimension < ndim
        self.ndim = ndim
        self.active_dimension = active_dimension    # first dimension is 0
        self.base_kernel_family = base_kernel_family
        
    def from_param_vector(self, params):
        return MaskKernel(self.ndim, self.active_dimension, self.base_kernel_family.from_param_vector(params))
    
    def num_params(self):
        return self.base_kernel_family.num_params()
    
    def pretty_print(self):
        return colored('Mask(%d, ' % self.active_dimension, self.depth()) + \
            self.base_kernel_family.pretty_print() + \
            colored(')', self.depth())
    
    def default(self):
        return MaskKernel(self.ndim, self.active_dimension, self.base_kernel_family.default())
    
    def __cmp__(self, other):
        assert isinstance(other, KernelFamily)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp((self.ndim, self.active_dimension, self.base_kernel_family),
                   (other.ndim, other.active_dimension, other.base_kernel_family))
        
    def depth(self):
        return self.base_kernel_family.depth() + 1
    
    
class MaskKernel(Kernel):
    def __init__(self, ndim, active_dimension, base_kernel):
        assert 0 <= active_dimension < ndim
        self.ndim = ndim
        self.active_dimension = active_dimension    # first dimension is 0
        self.base_kernel = base_kernel
        
    def copy(self):
        return MaskKernel(self.ndim, self.active_dimension, self.base_kernel.copy())
        
    def family(self):
        return MaskKernelFamily(self.ndim, self.active_dimension, self.base_kernel.family())
        
    def gpml_kernel_expression(self):
        dim_vec = np.zeros(self.ndim, dtype=int)
        dim_vec[self.active_dimension] = 1
        dim_vec_str = '[' + ' '.join(map(str, dim_vec)) + ']'
        return '{@covMask, {%s, %s}}' % (dim_vec_str, self.base_kernel.gpml_kernel_expression())
    
    def pretty_print(self):
        return colored('Mask(%d, ' % self.active_dimension, self.depth()) + \
            self.base_kernel.pretty_print() + \
            colored(')', self.depth())
    
    def param_vector(self):
        return self.base_kernel.param_vector()
    
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp((self.ndim, self.active_dimension, self.base_kernel),
                   (other.ndim, other.active_dimension, other.base_kernel))
        
    def depth(self):
        return self.base_kernel.depth() + 1
    

class SumKernelFamily(KernelFamily):
    def __init__(self, operands):
        self.operands = operands
        
    def from_param_vector(self, params):
        start = 0
        ops = []
        for e in self.operands:
            end = start + e.num_params()
            ops.append(e.from_param_vector(params[start:end]))
            start = end
        return SumKernel(ops)
    
    def num_params(self):
        return sum([e.num_params() for e in self.operands])
    
    def pretty_print(self):
        op = colored(' + ', self.depth())
        return colored('( ', self.depth()) + \
            op.join([e.pretty_print() for e in self.operands]) + \
            colored(' ) ', self.depth())
    
    def default(self):
        return SumKernel([op.default() for op in self.operands])
    
    def __cmp__(self, other):
        assert isinstance(other, KernelFamily)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp(self.operands, other.operands)
    
    def depth(self):
        return max([op.depth() for op in self.operands]) + 1

class SumKernel(Kernel):
    def __init__(self, operands):
        self.operands = operands
        
    def family(self):
        return SumKernelFamily([e.family() for e in self.operands])
        
    def pretty_print(self):
        op = colored(' + ', self.depth())
        return colored('( ', self.depth()) + \
            op.join([e.pretty_print() for e in self.operands]) + \
            colored(' ) ', self.depth())
    
    def gpml_kernel_expression(self):
        return '{@covSum, {%s}}' % ', '.join(e.gpml_kernel_expression() for e in self.operands)
    
    def copy(self):
        return SumKernel([e.copy() for e in self.operands])

    def param_vector(self):
        return np.concatenate([e.param_vector() for e in self.operands])
    
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp(self.operands, other.operands)
    
    def depth(self):
        return max([op.depth() for op in self.operands]) + 1
    
class ProductKernelFamily(KernelFamily):
    def __init__(self, operands):
        self.operands = operands
        
    def from_param_vector(self, params):
        start = 0
        ops = []
        for o in self.operands:
            end = start + o.num_params()
            ops.append(o.from_param_vector(params[start:end]))
            start = end
        return ProductKernel(ops)
    
    def num_params(self):
        return sum([e.num_params() for e in self.operands])
    
    def pretty_print(self):
        op = colored(' * ', self.depth())
        return colored('( ', self.depth()) + \
            op.join([e.pretty_print() for e in self.operands]) + \
            colored(' ) ', self.depth())
    
    def default(self):
        return ProductKernel([op.default() for op in self.operands])
    
    def __cmp__(self, other):
        assert isinstance(other, KernelFamily)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp(self.operands, other.operands)
    
    def depth(self):
        return max([op.depth() for op in self.operands]) + 1
    
class ProductKernel(Kernel):
    def __init__(self, operands):
        self.operands = operands
        
    def family(self):
        return ProductKernelFamily([e.family() for e in self.operands])
        
    def pretty_print(self):
        op = colored(' + ', self.depth())
        return colored('( ', self.depth()) + \
            op.join([e.pretty_print() for e in self.operands]) + \
            colored(' ) ', self.depth())
    
    def gpml_kernel_expression(self):
        return '{@covProd, {%s}}' % ', '.join(e.gpml_kernel_expression() for e in self.operands)
    
    def copy(self):
        return ProductKernel([e.copy() for e in self.operands])

    def param_vector(self):
        return np.concatenate([e.param_vector() for e in self.operands])
    
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return cmp(self.operands, other.operands)
    
    def depth(self):
        return max([op.depth() for op in self.operands]) + 1

            
def base_kernels(ndim=1):
    '''
    Generator of all base kernels for a certain dimensionality of data
    '''
    for dim in range(ndim):
        # Make up default arguments.
        yield MaskKernel(ndim, dim, SqExpKernel(0, 0))
        yield MaskKernel(ndim, dim, SqExpPeriodicKernel(0, 0, 0))
            
