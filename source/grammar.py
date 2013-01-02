import itertools

import flexiblekernel as fk


ONE_D_RULES = [('A', ('+', 'A', 'B'), {'A': 'any', 'B': 'base'}),        # replace with K plus a base kernel
               ('A', ('*', 'A', 'B'), {'A': 'any', 'B': 'base'}),        # replace with K times a base kernel
               ('A', 'B', {'A': 'base', 'B': 'base'}),                   # replace one base kernel with another
               ]

class OneDGrammar:
    def __init__(self):
        self.rules = ONE_D_RULES
    
    def type_matches(self, kernel, tp):
        if tp == 'any':
            return True
        elif tp == 'base':
            #### FIXME
            #return isinstance(tp, fk.SqExpKernel) or \
            #    isinstance(tp, fk.SqExpPeriodicKernel) or \
            #    isinstance(tp, fk.RQKernel)
            return isinstance(kernel, fk.BaseKernel)
        else:
            raise RuntimeError('Unknown type: %s' % tp)
    
    def list_options(self, tp):
        if tp == 'any':
            raise RuntimeError("Can't expand the 'any' type")
        elif tp == 'base':
            return list(fk.base_kernel_families())
            #return list(fk.test_kernel_families())
        else:
            raise RuntimeError('Unknown type: %s' % tp)
        
MULTI_D_RULES = [#('A', ('+', 'A', 'B'), {'A': '1d', 'B': 'base'}),
                 #('A', ('*', 'A', 'B'), {'A': '1d', 'B': 'base'}),
                 ('A', ('+', 'A', 'B'), {'A': 'multi', 'B': 'mask'}),
                 ('A', ('*', 'A', 'B'), {'A': 'multi', 'B': 'mask'}),
                 ('A', 'B', {'A': 'base', 'B': 'base'}),
                 ]
    
class MultiDGrammar:
    def __init__(self, ndim, debug=False):
        self.rules = MULTI_D_RULES
        self.ndim = ndim
        self.debug = debug
        
    def type_matches(self, kernel, tp):
        if tp == 'multi':
            if isinstance(kernel, fk.BaseKernel):
                return False
            elif isinstance(kernel, fk.MaskKernel):
                return True
            elif isinstance(kernel, fk.SumKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            elif isinstance(kernel, fk.ProductKernel):
                return all([self.type_matches(op, 'multi') for op in kernel.operands])
            else:
                raise RuntimeError('Invalid kernel: %s' % kernel.pretty_print())
        elif tp == '1d':
            if isinstance(kernel, fk.BaseKernel):
                return True
            elif isinstance(kernel, fk.MaskKernel):
                return False
            elif isinstance(kernel, fk.SumKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            elif isinstance(kernel, fk.ProductKernel):
                return all([self.type_matches(op, '1d') for op in kernel.operands])
            else:
                raise RuntimeError('Invalid kernel: %s' % kernel.pretty_print())
        elif tp == 'base':
            return isinstance(kernel, fk.BaseKernel)
        elif tp == 'mask':
            return isinstance(kernel, fk.MaskKernel)
        else:
            raise RuntimeError('Unknown type: %s' % tp)
        
    def list_options(self, tp):
        if tp in ['1d', 'multi']:
            raise RuntimeError("Can't expand the '%s' type" % tp)
        elif tp == 'base':
            if self.debug:
                return list(fk.test_kernel_families())
            else:
                return list(fk.base_kernel_families())
        elif tp == 'mask':
            result = []
            for d in range(self.ndim):
                if self.debug:
                    result += [fk.MaskKernel(self.ndim, d, fam_default) for fam_default in fk.test_kernel_families()]
                else:
                    result += [fk.MaskKernel(self.ndim, d, fam_default) for fam_default in fk.base_kernel_families()]
            return result
        else:
            raise RuntimeError('Unknown type: %s' % tp)
    
def replace_all(polish_expr, mapping):
    if type(polish_expr) == tuple:
        return tuple([replace_all(e, mapping) for e in polish_expr])
    elif type(polish_expr) == str:
        if polish_expr in mapping:
            return mapping[polish_expr].copy()
        else:
            return polish_expr
    else:
        assert isinstance(polish_expr, fk.Kernel)
        return polish_expr.copy()
    
def polish_to_kernel(polish_expr):
    if type(polish_expr) == tuple:
        if polish_expr[0] == '+':
            operands = [polish_to_kernel(e) for e in polish_expr[1:]]
            return fk.SumKernel(operands)
        elif polish_expr[0] == '*':
            operands = [polish_to_kernel(e) for e in polish_expr[1:]]
            return fk.ProductKernel(operands)
        else:
            raise RuntimeError('Unknown operator: %s' % polish_expr[0])
    else:
        assert isinstance(polish_expr, fk.Kernel)
        return polish_expr


def expand_single_tree(kernel, grammar):
    '''kernel should be a Kernel.'''
    assert isinstance(kernel, fk.Kernel)
    result = []
    for lhs, rhs, types in grammar.rules:
        if grammar.type_matches(kernel, types[lhs]):
            free_vars = types.keys()
            assert lhs in free_vars
            free_vars.remove(lhs)
            choices = itertools.product(*[grammar.list_options(types[v]) for v in free_vars])
            for c in choices:
                mapping = dict(zip(free_vars, c))
                mapping[lhs] = kernel
                full_polish = replace_all(rhs, mapping)
                result.append(polish_to_kernel(full_polish))
    return result

def expand(kernel, grammar):
    result = expand_single_tree(kernel, grammar)
    if isinstance(kernel, fk.BaseKernel):
        pass
    elif isinstance(kernel, fk.MaskKernel):
        result += [fk.MaskKernel(kernel.ndim, kernel.active_dimension, e)
                   for e in expand(kernel.base_kernel, grammar)]
    elif isinstance(kernel, fk.SumKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.SumKernel(new_ops))
    elif isinstance(kernel, fk.ProductKernel):
        for i, op in enumerate(kernel.operands):
            for e in expand(op, grammar):
                new_ops = kernel.operands[:i] + [e] + kernel.operands[i+1:]
                new_ops = [op.copy() for op in new_ops]
                result.append(fk.ProductKernel(new_ops))
    else:
        raise RuntimeError('Unknown kernel class:', kernel.__class__)
    return result

def canonical(kernel):
    '''Sorts a kernel tree into a canonical form.'''
    if isinstance(kernel, fk.BaseKernel):
        return kernel.copy()
    elif isinstance(kernel, fk.MaskKernel):
        return fk.MaskKernel(kernel.ndim, kernel.active_dimension, canonical(kernel.base_kernel))
    elif isinstance(kernel, fk.SumKernel):
        new_ops = []
        for op in kernel.operands:
            op_canon = canonical(op)
            if isinstance(op, fk.SumKernel):
                new_ops += op_canon.operands
            else:
                new_ops.append(op_canon)
        return fk.SumKernel(sorted(new_ops))
    elif isinstance(kernel, fk.ProductKernel):
        new_ops = []
        for op in kernel.operands:
            op_canon = canonical(op)
            if isinstance(op, fk.ProductKernel):
                new_ops += op_canon.operands
            else:
                new_ops.append(op_canon)
        return fk.ProductKernel(sorted(new_ops))
    else:
        raise RuntimeError('Unknown kernel class:', kernel.__class__)

def remove_duplicates(kernels):
    kernels = sorted(map(canonical, kernels))
    result = []
    curr = None
    for k in kernels:
        if curr is None or k != curr:
            result.append(k)
        curr = k
    return result
    




            
