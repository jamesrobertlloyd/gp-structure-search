import collections
import itertools
import numpy as np
nax = np.newaxis
import Image
import mkl_hack
import scipy.linalg


def _err_string(arr1, arr2):
    try:
        if np.allclose(arr1, arr2):
            return 'OK'
        elif arr1.shape == arr2.shape:
            return 'off by %s' % np.abs(arr1 - arr2).max()
        else:
            return 'incorrect shapes: %s and %s' % (arr1.shape, arr2.shape)
    except:
        return 'error comparing'

err_info = collections.defaultdict(list)
def set_err_info(key, info):
    err_info[key] = info

def summarize_error(key):
    """Print a helpful description of the reason a condition was not satisfied. Intended usage:
        assert pot1.allclose(pot2), summarize_error()"""
    if type(err_info[key]) == str:
        return '    ' + err_info[key]
    else:
        return '\n' + '\n'.join(['    %s: %s' % (name, err) for name, err in err_info[key]]) + '\n'


def broadcast(idx, shape):
    result = []
    for i, d in zip(idx, shape):
        if d == 1:
            result.append(0)
        else:
            result.append(i)
    return tuple(result)

def full_shape(shapes):
    """The shape of the full array that results from broadcasting the arrays of the given shapes."""
    #return tuple(np.array(shapes).max(0))
    temp = np.array(shapes)
    temp1 = np.where((temp==0).any(0), 0, temp.max(0))
    return tuple(temp1)


def array_map(fn, arrs, n):
    """Takes a list of arrays a_1, ..., a_n where the elements of the first n dimensions line up. For every possible
    index into the first n dimensions, apply fn to the corresponding slices, and combine the results into
    an n-dimensional array. Supports broadcasting but does not prepend 1's to the shapes."""
    # we shouldn't need a special case for n == 0, but NumPy complains about indexing into a zero-dimensional
    # array a using a[(Ellipsis,)].
    if n == 0:
        return fn(*arrs)
    
    full_shape = tuple(np.array([a.shape[:n] for a in arrs]).max(0))
    result = None
    for full_idx in itertools.product(*map(range, full_shape)):
        inputs = [a[broadcast(full_idx, a.shape[:n]) + (Ellipsis,)] for a in arrs]
        curr = fn(*inputs)
        
        if result is None:
            if type(curr) == tuple:
                result = tuple(np.zeros(full_shape + np.asarray(c).shape) for c in curr)
            else:
                result = np.zeros(full_shape + np.asarray(curr).shape)

        if type(curr) == tuple:
            for i, c in enumerate(curr):
                result[i][full_idx + (Ellipsis,)] = c
        else:
            result[full_idx + (Ellipsis,)] = curr
    return result

def extend_slice(slc, n):
    if not isinstance(slc, tuple):
        slc = (slc,)
    #if any([isinstance(s, np.ndarray) for s in slc]):
    #    raise NotImplementedError('Advanced slicing not implemented yet')
    return slc + (slice(None),) * n

def process_slice(slc, shape, n):
    """Takes a slice and returns the appropriate slice into an array that's being broadcast (i.e. by
    converting the appropriate entries to 0's and :'s."""
    if not isinstance(slc, tuple):
        slc = (slc,)
    slc = list(slc)
    ndim = len(shape) - n
    assert ndim >= 0
    shape_idx = 0
    for slice_idx, s in enumerate(slc):
        if s == nax:
            continue
        if shape[shape_idx] == 1:
            if type(s) == int:
                slc[slice_idx] = 0
            else:
                slc[slice_idx] = slice(None)
        shape_idx += 1
    if shape_idx != ndim:
        raise IndexError('Must have %d terms in the slice object' % ndim)
    return extend_slice(tuple(slc), n)

def my_sum(a, axis, count):
    """For an array a which might be broadcast, return the value of a.sum() were a to be expanded out in full."""
    if a.shape[axis] == count:
        return a.sum(axis)
    elif a.shape[axis] == 1:
        return count * a.sum(axis)
    else:
        raise IndexError('Cannot be broadcast: a.shape=%s, axis=%d, count=%d' % (a.shape, axis, count))
        
    

def match_shapes(arrs):
    """Prepend 1's to the shapes so that the dimensions line up."""
    #temp = [(name, np.asarray(a), deg) for name, a, deg in arrs]
    #ndim = max([a.ndim - deg for _, a, deg in arrs])

    temp = [a for name, a, deg in arrs]
    for i in range(len(temp)):
        if np.isscalar(temp[i]):
            temp[i] = np.array(temp[i])
    ndim = max([a.ndim - deg for a, (_, _, deg) in zip(temp, arrs)])

    prep_arrs = []
    for name, a, deg in arrs:
        if np.isscalar(a):
            a = np.asarray(a)
        if a.ndim < deg:
            raise RuntimeError('%s.ndim must be at least %d' % (name, deg))
        if a.ndim < ndim + deg:
            #a = a.reshape((1,) * (ndim + deg - a.ndim) + a.shape)
            slc = (nax,) * (ndim + deg - a.ndim) + (Ellipsis,)
            a = a[slc]
        prep_arrs.append(a)

    return prep_arrs
    
def lstsq(A, b):
    # do this rather than call lstsq to support efficient broadcasting
    P = array_map(np.linalg.pinv, [A], A.ndim - 2)
    return array_map(np.dot, [P, b], A.ndim - 2)

def dot(A, b):
    return array_map(np.dot, [A, b], A.ndim - 2)

def vdot(x, y):
    return (x*y).sum(-1)

def my_inv(A):
    """Compute the inverse of a symmetric positive definite matrix."""
    cho = scipy.linalg.flapack.dpotrf(A)
    choinv = scipy.linalg.flapack.dtrtri(cho[0])
    upper = scipy.linalg.flapack.dlauum(choinv[0])[0]

    # upper is the upper triangular entries of A^{-1}, so need to fill in the
    # lower triangular ones; unfortunately this has nontrivial overhead
    temp = np.diag(upper)
    return upper + upper.T - np.diag(temp)


def transp(A):
    return A.swapaxes(-2, -1)

def resize(arr, size):
    assert arr.ndim in [2, 3]
    if arr.ndim == 3:
        #return np.concatenate([shape_to_cons('**1', resize(arr[:,:,i], size))
        #                       for i in range(3)], axis=2)
        ans = np.concatenate([resize(arr[:,:,i], size)[:,:,nax] for i in range(3)], axis=2)
        return ans
    M, N = arr.shape
    assert arr.dtype in ['float64', 'float32']
    dtype = arr.dtype
    m, n = size
    if m is None:
        assert n is not None
        m = int(M * (float(n)/float(N)))
    if n is None:
        assert m is not None
        n = int(N * (float(m)/float(M)))

    result = np.array(Image.fromarray(arr.astype('float32'), 'F').resize((n, m), Image.ANTIALIAS),
                         dtype=dtype)

    return result
