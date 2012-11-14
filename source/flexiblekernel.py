'''
Created Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import numpy as np
try:
    import termcolor
    has_termcolor = True
except:
    has_termcolor = False

import config

PAREN_COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
CMP_TOLERANCE = np.log(1.01) # i.e. 1%

def paren_colors():
    if config.COLOR_SCHEME == 'dark':
        return ['red', 'green', 'cyan', 'magenta', 'yellow']
    elif config.COLOR_SCHEME == 'light':
        return ['blue', 'red', 'magenta', 'green', 'cyan']
    else:
        raise RuntimeError('Unknown color scheme: %s' % config.COLOR_SCHEME)

def colored(text, depth):
    if has_termcolor:
        colors = paren_colors()
        color = colors[depth % len(colors)]
        return termcolor.colored(text, color, attrs=['bold'])
    else:
        return text

class KernelFamily:
    pass

class Kernel:
    def __add__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, SumKernel):
            return SumKernel([self] + other.operands).copy()
        else:
            return SumKernel([self, other]).copy()
    
    def __mul__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, ProductKernel):
            return ProductKernel([self] + other.operands).copy()
        else:
            return ProductKernel([self, other])

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
        return colored('SE(ell=%1.1f, sf=%1.1f)' % (self.lengthscale, self.output_variance), self.depth())
    
    def latex_print(self):
        #return 'SE(\\ell=%1.1f, \\sigma=%1.1f)' % (self.lengthscale, self.output_variance)    
        #return 'SE(\\ell=%1.1f)' % self.lengthscale
        return 'SE'
        
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        max_diff = max(np.abs([self.lengthscale - other.lengthscale, self.output_variance - other.output_variance]))
        return max_diff > CMP_TOLERANCE
#        return cmp((self.lengthscale, self.output_variance), (other.lengthscale, other.output_variance))
    
    def depth(self):
        return 0


class SqExpPeriodicKernelFamily(BaseKernelFamily):
    def from_param_vector(self, params):
        output_variance, period, lengthscale = params
        return SqExpPeriodicKernel(output_variance, period, lengthscale)
    
    def num_params(self):
        return 3
    
    def pretty_print(self):
        return colored('PE', self.depth())
    
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
        return colored('PE(ell=%1.1f, p=%1.1f, sf=%1.1f)' % (self.lengthscale, self.period, self.output_variance),
                       self.depth())
        
    def latex_print(self):
        # return 'PE(\\ell=%1.1f, p=%1.1f, \\sigma=%1.1f)' % (self.lengthscale, self.period, self.output_variance)
        #return 'PE(p=%1.1f)' % self.period          
        return 'PE'
    
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        max_diff = max(np.abs([self.lengthscale - other.lengthscale, self.period - other.period, self.output_variance - other.output_variance]))
        return max_diff > CMP_TOLERANCE
#        return cmp((self.lengthscale, self.period, self.output_variance), 
#                   (other.lengthscale, other.period, other.output_variance))
        
    def depth(self):
        return 0
    

class RQKernelFamily(BaseKernelFamily):
    def from_param_vector(self, params):
        lengthscale, output_variance, alpha = params
        return RQKernel(lengthscale, output_variance, alpha)
    
    def num_params(self):
        return 3
    
    def pretty_print(self):
        return colored('RQ', self.depth())
    
    def default(self):
        return RQKernel(0., 0., 0.)
    
    def __cmp__(self, other):
        assert isinstance(other, KernelFamily)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        return 0
    
    def depth(self):
        return 0
    
    
class RQKernel(BaseKernel):
    def __init__(self, lengthscale, output_variance, alpha):
        self.lengthscale = lengthscale
        self.output_variance = output_variance
        self.alpha = alpha
        
    def family(self):
        return RQKernelFamily()
        
    def gpml_kernel_expression(self):
        return '@covRQiso'
    
    def english_name(self):
        return 'RQ'
    
    def param_vector(self):
        # order of args matches GPML
        return np.array([self.lengthscale, self.output_variance, self.alpha])

    def copy(self):
        return RQKernel(self.lengthscale, self.output_variance, self.alpha)
    
    def __repr__(self):
        return 'RQKernel(lengthscale=%f, output_variance=%f, alpha=%f)' % \
            (self.lengthscale, self.output_variance, self.alpha)
    
    def pretty_print(self):
        return colored('RQ(ell=%1.1f, sf=%1.1f, a=%1.1f)' % (self.lengthscale, self.output_variance, self.alpha),
                       self.depth())
        
    def latex_print(self):
        #return 'RQ(\\ell=%1.1f, \\alpha=%1.1f, \\sigma=%1.1f)' % (self.lengthscale, self.alpha, self.output_variance)
        #return 'RQ(\\ell=%1.1f)' % self.lengthscale
        return 'RQ'           
    
    def __cmp__(self, other):
        assert isinstance(other, Kernel)
        if cmp(self.__class__, other.__class__):
            return cmp(self.__class__, other.__class__)
        max_diff = max(np.abs([self.lengthscale - other.lengthscale, self.output_variance - other.output_variance, self.alpha - other.alpha]))
        return max_diff > CMP_TOLERANCE
#        return cmp((self.lengthscale, self.output_variance, self.alpha), 
#                   (other.lengthscale, other.output_variance, other.alpha))
        
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
        #return colored('Mask(%d, ' % self.active_dimension, self.depth()) + \
        return colored('M(%d, ' % self.active_dimension, self.depth()) + \
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
        return colored('M(%d, ' % self.active_dimension, self.depth()) + \
            self.base_kernel.pretty_print() + \
            colored(')', self.depth())
            
    def latex_print(self):
        #return 'M_%d \\left(' % self.active_dimension + self.base_kernel.latex_print() + '\\right)'                 
        return self.base_kernel.latex_print() + '_%d' % self.active_dimension
            
    def __repr__(self):
        return 'MaskKernel(ndim=%d, active_dimension=%d, base_kernel=%s)' % \
            (self.ndim, self.active_dimension, self.base_kernel.__repr__())            
    
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
        #### Should this call the family method?
        op = colored(' + ', self.depth())
        return colored('( ', self.depth()) + \
            op.join([e.pretty_print() for e in self.operands]) + \
            colored(' ) ', self.depth())
            
    def latex_print(self):
        return '\\left( ' + ' + '.join([e.latex_print() for e in self.operands]) + ' \\right)'            
            
    def __repr__(self):
        return 'SumKernel(%s)' % \
            ('[ ' + ', '.join([o.__repr__() for o in self.operands]) + ' ]')                
    
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
    
    def __add__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, SumKernel):
            return SumKernel(self.operands + other.operands).copy()
        else:
            return SumKernel(self.operands + [other]).copy()
    
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
        op = colored(' x ', self.depth())
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
        #### Should this call the family method?
        op = colored(' x ', self.depth())
        return colored('( ', self.depth()) + \
            op.join([e.pretty_print() for e in self.operands]) + \
            colored(' ) ', self.depth())

    def latex_print(self):
        #return '\\left( ' + ' \\times '.join([e.latex_print() for e in self.operands]) + ' \\right)'
        # Don't need brackets for product, order of operations is unambiguous, I think.
        return ' \\times '.join([e.latex_print() for e in self.operands])
            
    def __repr__(self):
        return 'ProductKernel(%s)' % \
            ('[ ' + ', '.join([o.__repr__() for o in self.operands]) + ' ]')              
    
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
    
    def __mul__(self, other):
        assert isinstance(other, Kernel)
        if isinstance(other, ProductKernel):
            return ProductKernel(self.operands + other.operands).copy()
        else:
            return ProductKernel(self.operands + [other]).copy()

            
def base_kernels(ndim=1):
    '''
    Generator of all base kernels for a certain dimensionality of data
    '''
    for dim in range(ndim):
        # Make up default arguments.
        yield MaskKernel(ndim, dim, SqExpKernel(0, 0))
        yield MaskKernel(ndim, dim, SqExpPeriodicKernel(0, 0, 0))
        yield MaskKernel(ndim, dim, RQKernel(0, 0, 0))
        
            

def Carls_Mauna_kernel():
    '''
    This kernel described in pages 120-122 of "Gaussian Processes for Machine Learning.
    This model was learnt on the mauna dataset up to 2003.
    
    The reported nll in the book for this dataset is 108.5
    '''
    theta_1 = np.log(66.)  # ppm, sf of SE1 = magnitude of long term trend
    theta_2 = np.log(67.)  # years, ell of SE1 = lengthscale of long term trend
    theta_6 = np.log(0.66)  # ppm, sf of RQ = magnitude of med term trend
    theta_7 = np.log(1.2)  # years, ell of RQ = lengthscale of med term trend
    theta_8 = np.log(0.78) # alpha of RQ
    theta_3 = np.log(2.4) # ppm, sf of periodic * SE
    theta_4 = np.log(90.) # years, lengthscale of SE of periodic*SE
    theta_5 = np.log(1.3) # smoothness of periodic
    theta_9 = np.log(0.18) # ppm, amplitude of SE_noise
    theta_10 = np.log(1.6/12.0) # years (originally months), lengthscale of SE_noise
    theta_11 = np.log(0.19) # ppm, amplitude of independent noise
    
    kernel = SqExpKernel(output_variance=theta_1, lengthscale=theta_2) \
           + SqExpKernel(output_variance=theta_3, lengthscale=theta_4) * SqExpPeriodicKernel(output_variance=0, period=0, lengthscale=theta_5) \
           + RQKernel(lengthscale=theta_7, output_variance=theta_6, alpha=theta_8) \
           + SqExpKernel(output_variance=theta_9, lengthscale=theta_10)
    
    return kernel

def repr_string_to_kernel(string):
    '''This is defined in this module so that all the kernel class names
    don't have to have the module name in front of them.'''
    return eval(string)

 
