'''
Created Nov 2012

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
          David Duvenaud (dkd23@cam.ac.uk)
          Roger Grosse (rgrosse@mit.edu)
'''

import numpy as np

#class KernelNames:
#    '''
#    A kernel name enumeration
#    '''
#    SqExp = 1
#    SqExpPeriodic = 2
#    Matern = 3
#    MaternPeriodic = 4

class BaseKernel(object):
    '''
    A kernel object that knows what it is and how to write itself down etc.
    
    Base kernels don't have any operators.
    '''
    #### HACK - fixmeEither a kernel or an operator with a list of expressions
    def polish_expression(self):
        return '%s_%s' % (self.english_name(), ','.join(str(i+1) for (i, d) in enumerate(self.gpml_dimension_vector()) if d))
        
    def gpml_kernel_expression(self):
        # todo: check if active_dimensions exists
        return '{@covMask, {%s, %s}}' % (self.gpml_dimension_vector(), self.gpml_name())
    
    def gpml_param_expression(self):
        return '; '.join(str(f) for f in self.param_vector())
    


class SqExpKernel(BaseKernel):
    def __init__(self, ndim, active_dimension, lengthscale, output_variance):
        assert 0 <= active_dimension < ndim
        self.ndim = ndim
        self.active_dimension = active_dimension    # first dimension is 0
        self.output_variance = output_variance
        self.lengthscale = lengthscale
        
    def gpml_name(self):
        return '@covSEiso'
    
    def english_name(self):
        return 'SqExp'
    
    def param_vector(self):
        # order of args matches GPML
        return np.array([self.lengthscale, self.output_variance])
    
    def gpml_dimension_vector(self):
        result = np.zeros(self.ndim, dtype=int)
        result[self.active_dimension] = 1
        return result
    
    def copy(self):
        return SqExpKernel(self.ndim, self.active_dimension, self.lengthscale, self.output_variance)
    
    def __repr__(self):
        return 'SqExpKernel(ndim=%d, active_dimension=%d, lengthscale=%f, output_variance=%f)' % \
            (self.ndim, self.active_dimension, self.lengthscale, self.output_variance)
    
    @staticmethod
    def from_param_vector(self, ndim, active_dimension, params):
        output_variance, lengthscale = params
        return SqExpKernel(ndim, active_dimension, output_variance, lengthscale)
        
    
class SqExpPeriodicKernel(BaseKernel):
    def __init__(self, ndim, active_dimension, lengthscale, period, output_variance):
        assert 0 <= active_dimension < ndim
        self.ndim = ndim
        self.active_dimension = active_dimension    # first dimension is 0
        self.lengthscale = lengthscale
        self.period = period
        self.output_variance = output_variance
        
    def gpml_name(self):
        return '@covPeriodic'
    
    def english_name(self):
        return 'Periodic'
    
    def param_vector(self):
        # order of args matches GPML
        return np.array([self.lengthscale, self.period, self.output_variance])
    
    def gpml_dimension_vector(self):
        result = np.zeros(self.ndim, dtype=int)
        result[self.active_dimension] = 1
        return result
    
    def copy(self):
        return SqExpKernel(self.ndim, self.active_dimension, self.lengthscale, self.period, self.output_variance)
    
    def __repr__(self):
        return 'SqExpPeriodicKernel(ndim=%d, active_dimension=%d, lengthscale=%f, period=%f, output_variance=%f)' % \
            (self.ndim, self.active_dimension, self.lengthscale, self.period, self.output_variance)
    
    @staticmethod
    def from_param_vector(self, ndim, active_dimension, params):
        output_variance, period, lengthscale = params
        return SqExpPeriodicKernel(ndim, active_dimension, output_variance, period, lengthscale)
    
    
    
    
    
            
def base_kernels(ndim=1):
    '''
    Generator of all base kernels for a certain dimensionality of data
    '''
    for dim in range(ndim):
        # Make up default arguments.
        yield SqExpKernel(ndim, dim, 0, 0)
        yield SqExpPeriodicKernel(ndim, dim, 0, 0, 0)
            
class CompoundKernel(object):
    '''
    Defines a tree of operators and base kernels, as well as their parameters.
    '''
    
    def __init__(self, kernel=[], operator='', operands=[]):
        '''
        Constructor
        '''
        self.kernel = kernel
        self.operator = operator
        self.operands = operands
        # Check that we are either a kernel or a higher level expression
        self.__param_check()    
    
    def __param_check(self):
        if (self.kernel != []) and ((len(self.operator) > 0) or (len(self.operands) > 0)):
            self.operator = ''
            self.operands = []
            raise Exception('Contradictory parameters, ignoring operator and operands')
        
    def gpml_operator(self):
        if self.operator == '+':
            return '@covSum'
        elif self.operator == 'x':
            return '@covProd'
        else:
            raise Exception('Unrecognised operator in CompoundKernel.gpml_operator')
        
    def infix_expression(self):
        # This will be able to call kernel.polish_expression but will need to shunt around operators
        pass
        
    def polish_expression(self):
        #### TODO - turn this test into a method
        if len(self.operands) == 0:
            # We are a kernel/leaf, print directly
            return self.kernel.polish_expression()
        else:
            # We are an operator/non-terminal, recurse
            return '(%s %s)' % (self.operator, ' '.join(e.polish_expression() for e in self.operands))
        
    def gpml_kernel_expression(self):
        if len(self.operands) == 0:
            # We are a kernel/leaf, print directly
            return self.kernel.gpml_kernel_expression()
        else:
            # We are an operator/non-terminal, recurse
            return '{%s, {%s}}' % (self.gpml_operator(), ', '.join(e.gpml_kernel_expression() for e in self.operands))
        
    def gpml_param_expression(self):
        if len(self.operands) == 0:
            # We are a kernel/leaf, print directly
            return self.kernel.gpml_param_expression()
        else:
            # We are an operator/non-terminal, recurse
            return '; '.join(e.gpml_param_expression() for e in self.operands)
        
    def copy(self):
        # I think this will be useful for generating new expressions
        if len(self.operands) == 0:
            kernel_copy = self.kernel.copy()
            return CompoundKernel(kernel=kernel_copy)
        else:
            return CompoundKernel(operator=self.operator, operands=[e.copy() for e in self.operands])
        
    #### FIXME d=1 is a hack
    def expand(self, d=1, operator=''):
        '''
        Return a generator of potential expressions one step away from the current expression
        '''
        if len(self.operands) == 0:
            # Terminal - cannot be expanded - NOT TRUE!
            if operator == '':
                raise Exception ('This should not happen!')
            else:
                for k in base_kernels(d):
                    if not ((operator == 'x') and (self.kernel.name == k.name) and (self.kernel.active_dimensions == k.active_dimensions)):   
                        yield CompoundKernel(operator=operator, operands=[self.copy(), CompoundKernel(k)])
            
        elif self.operator == '+':
            for (i, e) in enumerate(self.operands):
                for f in e.expand(d, 'x'):
                    copy = self.copy()
                    copy.operands[i] = f
                    yield copy      
            for k in base_kernels(d):
                copy = self.copy()
                copy.operands.append(CompoundKernel(k))
                yield copy
            
        elif self.operator == 'x':
            for (i, e) in enumerate(self.operands):
                for f in e.expand(d, '+'):
                    copy = self.copy()
                    copy.operands[i] = f
                    yield copy    
            for k in base_kernels(d):
                redundant = False
                for e in self.operands:
                    if (len(e.operands) == 0) and (e.kernel.name == k.name) and (e.kernel.active_dimensions == k.active_dimensions):
                        redundant = True
                if not redundant:
                    copy = self.copy()
                    copy.operands.append(CompoundKernel(k))
                    yield copy
