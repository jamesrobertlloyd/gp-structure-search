'''
Created on Nov 2012

@author: James Robert Lloyd
'''

import flexiblekernel

def kernel_test():
    k = flexiblekernel.Kernel(active_dimensions=[0,0,1,1,1])
    print k.gpml_kernel_expression()
    print k.polish_expression()
    print '[%s]' % k.gpml_param_expression()
    print 'kernel_test complete'
    
def expression_test():
    k1 = flexiblekernel.KernelGrammarExpression(flexiblekernel.Kernel(active_dimensions=[1,0,0,0]))
    k2 = flexiblekernel.KernelGrammarExpression(flexiblekernel.Kernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[0,1,0,0], params=[1,1,1]))
    k3 = flexiblekernel.KernelGrammarExpression(flexiblekernel.Kernel(active_dimensions=[0,0,1,0], params=[3,4]))
    k4 = flexiblekernel.KernelGrammarExpression(flexiblekernel.Kernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[0,0,0,1], params=[2,2,2]))
    f = flexiblekernel.KernelGrammarExpression(operator = '*', operands = [k3, k4])
    e = flexiblekernel.KernelGrammarExpression(operator = '+', operands = [k1, k2, f])
    print e.polish_expression()
    print e.gpml_kernel_expression()
    print '[%s]' % e.gpml_param_expression()
    print 'expression_test complete'

def base_kernel_test():
    print [k.polish_expression() for k in flexiblekernel.base_kernels(5)]
    print 'base_kernel_test complete'
    
def expand_test():
    k1 = flexiblekernel.KernelGrammarExpression(flexiblekernel.Kernel(active_dimensions=[1,0,0,0]))
    k2 = flexiblekernel.KernelGrammarExpression(flexiblekernel.Kernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[0,1,0,0], params=[1,1,1]))
    k3 = flexiblekernel.KernelGrammarExpression(flexiblekernel.Kernel(active_dimensions=[0,0,1,0], params=[3,4]))
    k4 = flexiblekernel.KernelGrammarExpression(flexiblekernel.Kernel(name=flexiblekernel.KernelNames.SqExpPeriodic, active_dimensions=[0,0,0,1], params=[2,2,2]))
    f = flexiblekernel.KernelGrammarExpression(operator = 'x', operands = [k3, k4])
    e = flexiblekernel.KernelGrammarExpression(operator = '+', operands = [k1, k2, f])
    #e = flexiblekernel.KernelGrammarExpression(operator = '+', operands = [k1, k2])
    print e.polish_expression()
    print ''
    for f in e.expand(4):
        print f.gpml_kernel_expression()
    print 'expand_test complete'

if __name__ == '__main__':
    #kernel_test()
    #expression_test()
    #base_kernel_test()
    expand_test()