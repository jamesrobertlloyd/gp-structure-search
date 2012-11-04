import itertools

import flexiblekernel as fk


RULES = [('A', ('+', 'A', 'B'), {'A': 'any', 'B': 'base'}),        # replace with K plus a base kernel
         ('A', ('*', 'A', 'B'), {'A': 'any', 'B': 'base'}),        # replace with K times a base kernel
         ('A', 'B', {'A': 'base', 'B': 'base'}),                   # replace one base kernel with another
         ]

def type_matches(kernel, tp):
    if tp == 'any':
        return True
    elif tp == 'base':
        return isinstance(tp, fk.SqExpKernelFamily) or \
            isinstance(tp, fk.SqExpPeriodicKernelFamily)
    else:
        raise RuntimeError('Unknown type: %s' % tp)
    
def list_options(tp):
    if tp == 'any':
        raise RuntimeError("Can't expand the 'any' type")
    elif tp == 'base':
        return [fk.SqExpKernelFamily(), fk.SqExpPeriodicKernelFamily()]
    else:
        raise RuntimeError('Unknown type: %s' % tp)
    
def replace_all(polish_expr, mapping):
    if type(polish_expr) == tuple:
        return tuple([replace_all(e, mapping) for e in polish_expr])
    elif type(polish_expr) == str:
        if polish_expr in mapping:
            return mapping[polish_expr]
        else:
            return polish_expr
    else:   # kernel family class, probably
        return polish_expr
    
def polish_to_kernel(polish_expr):
    if type(polish_expr) == tuple:
        if polish_expr[0] == '+':
            operands = [polish_to_kernel(e) for e in polish_expr[1:]]
            return fk.SumKernelFamily(operands)
        elif polish_expr[0] == '*':
            operands = [polish_to_kernel(e) for e in polish_expr[1:]]
            return fk.ProductKernelFamily(operands)
        else:
            raise RuntimeError('Unknown operator: %s' % polish_expr[0])
    else:   # kernel family class
        return polish_expr


def expand(kernel, rules):
    result = []
    for lhs, rhs, types in rules:
        if type_matches(kernel, types[lhs]):
            free_vars = types.keys()
            assert lhs in free_vars
            free_vars.remove(lhs)
            choices = itertools.product([list_options(types[v]) for v in free_vars])
            for c in choices:
                mapping = dict(zip(free_vars, c))
                mapping[lhs] = kernel
                full_polish = replace_all(rhs, mapping)
                result.append(polish_to_kernel(full_polish))
    return result
            