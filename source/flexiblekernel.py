'''
Created Nov 2012

@author: James Robert Lloyd (jrl44@cam.ac.uk)
         David Duvenaud (dkd23@cam.ac.uk)
'''

class KernelNames:
    '''
    A kernel name enumeration
    '''
    SqExp = 1
    SqExpPeriodic = 2

class BaseKernel(object):
    '''
    A kernel object that knows what it is and how to write itself down etc.
    '''
    
    def n_params(self):
        if self.name == KernelNames.SqExp:
            return 1 + sum(self.active_dimensions)
        elif self.name == KernelNames.SqExpPeriodic:
            return 2 + sum(self.active_dimensions)
        else:
            raise Exception('Unrecognised kernel name in BaseKernel.n_params : %d' % self.name)
    
    def gpml_name(self):
        if self.name == KernelNames.SqExp:
            return '@covSEiso'
        elif self.name == KernelNames.SqExpPeriodic:
            return '@covPeriodic'
        else:
            raise Exception('Unrecognised kernel name in BaseKernel.gpml_name : %d' % self.name)
    
    def english_name(self):
        if self.name == KernelNames.SqExp:
            return 'SqExp'
        elif self.name == KernelNames.SqExpPeriodic:
            return 'SqExpPer'
        else:
            raise Exception('Unrecognised kernel name in BaseKernel.english_name : %d' % self.name)
        
    #### HACK - fixme
    def polish_expression(self):
        return '%s_%s' % (self.english_name(), ','.join(str(i+1) for (i, d) in enumerate(self.active_dimensions) if d))
        
    def gpml_kernel_expression(self):
        return '{@covMask, {%s, %s}}' % (self.active_dimensions, self.gpml_name())
    
    def gpml_param_expression(self):
        print self.params
        return '; '.join(str(f) for f in self.params)
    
    def clone(self):
        # I think this will be useful for generating new expressions
        return BaseKernel(self.name, self.active_dimensions[:], self.params[:])

    def __init__(self, name=KernelNames.SqExp, active_dimensions=[1], params=[]):
        '''
        Constructor
        '''
        self.name = name
        self.active_dimensions = active_dimensions
        self.params = params
        # Now make sure we have the correct number of parameters
        if len(params) != self.n_params():
            if len(self.params) > 0:
                # Maybe a default value was passed in
                self.params = [self.params[0]] * self.n_params()
            else:
                # Otherwise use the generic default of log(1)
                self.params = [0] * self.n_params()
            
def base_kernels(d=1):
    '''
    Generator of all base kernels for a certain dimensionality of data
    '''
    for dim in range(d):
        active_dimensions = [0] * d
        active_dimensions[dim] = 1
        yield BaseKernel(KernelNames.SqExp, active_dimensions, [0])
        yield BaseKernel(KernelNames.SqExpPeriodic, active_dimensions, [0])
            
class CompoundKernel(object):
    '''
    Either a kernel or an operator with a list of expressions
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
        
    def clone(self):
        # I think this will be useful for generating new expressions
        if len(self.operands) == 0:
            kernel_copy = self.kernel.clone()
            return CompoundKernel(kernel=kernel_copy)
        else:
            return CompoundKernel(operator=self.operator, operands=[e.clone() for e in self.operands])
        
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
                        yield CompoundKernel(operator=operator, operands=[self.clone(), CompoundKernel(k)])
            
        elif self.operator == '+':
            for (i, e) in enumerate(self.operands):
                for f in e.expand(d, 'x'):
                    copy = self.clone()
                    copy.operands[i] = f
                    yield copy      
            for k in base_kernels(d):
                copy = self.clone()
                copy.operands.append(CompoundKernel(k))
                yield copy
            
        elif self.operator == 'x':
            for (i, e) in enumerate(self.operands):
                for f in e.expand(d, '+'):
                    copy = self.clone()
                    copy.operands[i] = f
                    yield copy    
            for k in base_kernels(d):
                redundant = False
                for e in self.operands:
                    if (len(e.operands) == 0) and (e.kernel.name == k.name) and (e.kernel.active_dimensions == k.active_dimensions):
                        redundant = True
                if not redundant:
                    copy = self.clone()
                    copy.operands.append(CompoundKernel(k))
                    yield copy
