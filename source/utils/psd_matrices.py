import itertools
import numpy as np
nax = np.newaxis

import gaussians
from misc import array_map, my_inv, full_shape, broadcast, dot, process_slice, match_shapes, _err_string, set_err_info, transp
import scipy.linalg

class BaseMatrix:
    def __init__(self):
        self.mutable = False

    def set_mutable(self, m):
        self.mutable = m
        
    def __rmul__(self, other):
        return self * other

    def allclose(self, other):
        self, other = self.full(), other.full()
        es = _err_string(self._S, other._S)
        set_err_info('psd_matrices', [('S', es)])
        return np.allclose(self._S, other._S)

    def __getitem__(self, slc):
        return self.__slice__(slc)

    def __setitem__(self, slc, other):
        return self.__setslice__(slc, other)

class FullMatrix(BaseMatrix):
    def __init__(self, S):
        BaseMatrix.__init__(self)
        self._S = S
        self.shape = S.shape[:-2]
        self.ndim = len(self.shape)
        self.dim = S.shape[-1]
        self.shape_str = 'S=%s' % str(S.shape)

    def full(self):
        return self

    def copy(self):
        return FullMatrix(self._S.copy())

    def elt(self, i, j):
        return self._S[..., i, j]

    def col(self, j):
        return self._S[..., :, j]

    def __slice__(self, slc):
        S_slc = process_slice(slc, self._S.shape, 2)
        return FullMatrix(self._S[S_slc])

    def __setslice__(self, slc, other):
        if not self.mutable:
            raise RuntimeError('Attempt to modify an immutable matrix')
        S_slc = process_slice(slc, self._S.shape, 2)
        self._S[S_slc] += other.full()._S

    def dot(self, x):
        #return array_map(np.dot, [self._S, x], self.ndim)
        return dot(self._S, x)

    def qform(self, x):
        #temp = array_map(np.dot, [self._S, x], self.ndim)
        temp = dot(self._S, x)
        return (temp * x).sum(-1)

    def pinv(self):
        try:
            return FullMatrix(array_map(my_inv, [self._S], self.ndim))
        except np.linalg.LinAlgError:
            return FullMatrix(array_map(np.linalg.pinv, [self._S], self.ndim))

    def inv(self):
        return FullMatrix(array_map(my_inv, [self._S], self.ndim))

    def __add__(self, other):
        other = other.full()
        return FullMatrix(self._S + other._S)

    def __sub__(self, other):
        other = other.full()
        return FullMatrix(self._S - other._S)

    def __mul__(self, other):
        other = np.array(other)
        return FullMatrix(self._S * other[..., nax, nax])

    def sum(self, axis):
        assert axis >= 0
        return FullMatrix(self._S.sum(axis))

    def logdet(self):
        _, ld = array_map(np.linalg.slogdet, [self._S], self.ndim)
        return ld

    def alat(self, A):
        return FullMatrix(dot(A, dot(self._S, transp(A))))

    def rescale(self, a):
        slc = (nax,) * len(self.shape) + (slice(None), slice(None))
        return self.alat(a * np.eye(self.dim)[slc])

    def conv(self, other):
        other = other.full()
        P = array_map(my_inv, [self._S + other._S], self.ndim)
        return FullMatrix(dot(self._S, dot(P, other._S)))

    def sqrt_dot(self, x):
        L = array_map(np.linalg.cholesky, [self._S + 1e-10 * np.eye(self.dim)], self.ndim)
        return dot(L, x)

    def add_dummy_dimension(self):
        S = np.zeros(self.shape + (self.dim+1, self.dim+1))
        S[..., 1:, 1:] = self._S
        return FullMatrix(S)

    def to_eig(self):
        d, Q = array_map(np.linalg.eigh, [self._S], self.ndim)
        return FixedEigMatrix(d, Q, 0)


    @staticmethod
    def random(shape, dim, rank=None):
        if rank is None:
            rank = dim
        A = np.random.normal(size=shape + (dim, rank))
        S = dot(A, transp(A))
        return FullMatrix(S)


class DiagonalMatrix(BaseMatrix):
    def __init__(self, s):
        BaseMatrix.__init__(self)
        s = np.asarray(s)
        self._s = s
        self.shape = s.shape[:-1]
        self.ndim = len(self.shape)
        self.dim = s.shape[-1]
        self.shape_str = 's=%s' % str(s.shape)

    def full(self):
        S = array_map(np.diag, [self._s], self.ndim)
        return FullMatrix(S)

    def copy(self):
        return DiagonalMatrix(self._s.copy())

    def elt(self, i, j):
        if i == j:
            return self._s[..., i]
        else:
            return np.zeros(self.shape)

    def col(self, j):
        result = np.zeros(self.shape + (self.dim,))
        result[..., j] = self._s[..., j]
        return result

    def __slice__(self, slc):
        slc = process_slice(slc, self._s.shape, 1)
        return DiagonalMatrix(self._s[slc])

    def __setslice__(self, slc, other):
        if not self.mutable:
            raise RuntimeError('Attempt to modify an immutable matrix')
        if isinstance(other, DiagonalMatrix):
            slc = process_slice(slc, self._s.shape, 1)
            self._s[slc] = other._s
        else:
            raise RuntimeError('Cannot assign a DiagonalMatrix to a %s' % other.__class__)

    def dot(self, x):
        return self._s * x

    def qform(self, x):
        return (self._s * x**2).sum(-1)

    def pinv(self):
        return DiagonalMatrix(np.where(self._s > 0., 1. / self._s, 0.))

    def inv(self):
        return DiagonalMatrix(1. / self._s)

    def __add__(self, other):
        if isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self._s + other._s)
        elif isinstance(other, EyeMatrix):
            return DiagonalMatrix(self._s + other._s[..., nax])
        else:
            return self.full() + other

    def __sub__(self, other):
        return self + other * -1

    def __mul__(self, other):
        return DiagonalMatrix(self._s * other[..., nax])

    def sum(self, axis):
        assert axis >= 0
        return DiagonalMatrix(self._s.sum(axis))

    def logdet(self):
        return np.log(self._s).sum(-1)

    def alat(self, A):
        return self.full().alat(A)

    def rescale(self, a):
        a = np.asarray(a)
        return DiagonalMatrix(a[..., nax] ** 2 * self._s)

    def conv(self, other):
        if isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(1. / (1. / self._s + 1. / other._s))
        elif isinstance(other, EyeMatrix):
            return DiagonalMatrix(1. / (1. / self._s + 1. / other._s[..., nax]))
        else:
            return self.full().conv(other)

    def sqrt_dot(self, x):
        return np.sqrt(self._s) * x

    def add_dummy_dimension(self):
        return self.full().add_dummy_dimension()

    def to_eig(self):
        return self.full().to_eig()

    @staticmethod
    def random(shape, dim):
        s = np.random.gamma(1., 1., size=shape+(dim,))
        return DiagonalMatrix(s)
    

class EyeMatrix(BaseMatrix):
    def __init__(self, s, dim):
        BaseMatrix.__init__(self)
        s = np.asarray(s)
        self._s = s
        self.shape = s.shape
        self.ndim = len(self.shape)
        self.dim = dim
        self.shape_str = 's=%s' % str(s.shape)

    def full(self):
        return FullMatrix(self._s[..., nax, nax] * np.eye(self.dim))

    def copy(self):
        return EyeMatrix(self._s.copy(), self.dim)

    def elt(self, i, j):
        if i == j:
            return self._s
        else:
            return np.zeros(self.shape)

    def col(self, j):
        result = np.zeros(self.shape + (self.dim,))
        result[..., j] = self._s
        return result

    def __slice__(self, slc):
        return EyeMatrix(self._s[slc], self.dim)

    def __setslice__(self, slc, other):
        if not self.mutable:
            raise RuntimeError('Attempt to modify an immutable matrix')
        if isinstance(other, EyeMatrix):
            self._s[slc] = other._s
        else:
            raise RuntimeError('Cannot assign an EyeMatrix to a %s' % other.__class__)

    def dot(self, x):
        return self._s[..., nax] * x

    def qform(self, x):
        return (x ** 2).sum(-1) * self._s

    def pinv(self):
        return EyeMatrix(np.where(self._s > 0., 1. / self._s, 0.), self.dim)

    def inv(self):
        return EyeMatrix(1. / self._s, self.dim)

    def __add__(self, other):
        if isinstance(other, EyeMatrix):
            return EyeMatrix(self._s + other._s, self.dim)
        elif isinstance(other, FixedEigMatrix):
            return other + self
        elif isinstance(other, EigMatrix):
            return other + self
        elif isinstance(other, DiagonalMatrix):
            return other + self
        else:
            return self.full() + other

    def __sub__(self, other):
        return self + other * -1

    def __mul__(self, other):
        return EyeMatrix(self._s * other, self.dim)

    def sum(self, axis):
        return EyeMatrix(self._s.sum(axis), self.dim)

    def logdet(self):
        return self.dim * np.log(self._s)

    def alat(self, A):
        ## if A.shape[-1] == 1:
        ##     assert self.dim == 1
        ##     A_ = A[..., :, 0]
        ##     Q = A_ / np.sqrt((A_ ** 2).sum(-1))[..., nax]
        ##     d = (A_ ** 2).sum(-1) * self._s
        ##     return FixedEigMatrix(d[..., nax], Q[..., nax], 0.)
        
        return FullMatrix(dot(A, transp(A)) * self._s[..., nax, nax])

    def rescale(self, a):
        return EyeMatrix(a**2 * self._s, self.dim)
    

    def conv(self, other):
        if isinstance(other, EyeMatrix):
            return EyeMatrix(1. / (1. / self._s + 1. / other._s), self.dim)
        elif isinstance(other, EigMatrix):
            return other.conv(self)
        elif isinstance(other, FixedEigMatrix):
            return other.conv(self)
        else:
            return self.full().conv(other)

    def sqrt_dot(self, x):
        return np.sqrt(self._s)[..., nax] * x

    def add_dummy_dimension(self):
        return self.full().add_dummy_dimension()

    def to_eig(self):
        return self.full().to_eig()

    @staticmethod
    def random(shape, dim):
        s = np.random.gamma(1., 1., size=shape)
        return EyeMatrix(s, dim)





def _x_QDQ_x(Q, d, x):
    fsh = full_shape([x.shape[:-1], Q.shape, d.shape])
    prod = np.zeros(fsh)
    for full_idx in itertools.product(*map(range, fsh)):
        Q_idx = broadcast(full_idx, Q.shape)
        d_idx = broadcast(full_idx, d.shape)
        x_idx = broadcast(full_idx, x.shape[:-1])
        curr_d, curr_Q = d[d_idx], Q[Q_idx]
        curr_x = x[x_idx + (slice(None),)]
        curr_QTx = np.dot(curr_Q.T, curr_x)
        prod[full_idx] = (curr_d * curr_QTx**2).sum()
    return prod

def _QDQ_x(Q, d, x):
    fsh = full_shape([x.shape[:-1], Q.shape, d.shape])
    prod = np.zeros(fsh + (x.shape[-1],))
    for full_idx in itertools.product(*map(range, fsh)):
        Q_idx = broadcast(full_idx, Q.shape)
        d_idx = broadcast(full_idx, d.shape)
        x_idx = broadcast(full_idx, x.shape[:-1])
        curr_d, curr_Q = d[d_idx], Q[Q_idx]
        curr_x = x[x_idx + (slice(None),)]
        curr_QTx = np.dot(curr_Q.T, curr_x)
        prod[full_idx + (slice(None),)] = np.dot(curr_Q, curr_d * curr_QTx)
    return prod


class EigMatrix(BaseMatrix):
    def __init__(self, d, Q, s_perp, dim):
        BaseMatrix.__init__(self)
        d, Q, s_perp = match_shapes([('d', d, 0), ('Q', Q, 0), ('s_perp', s_perp, 0)])
        self._d = d
        self._Q = Q
        self.dim = dim
        self.ndim = d.ndim
        self._s_perp = s_perp
        self.shape = full_shape([d.shape, Q.shape, s_perp.shape])
        self.shape_str = 'd=%s Q=%s s_perp=%s' % (d.shape, Q.shape, s_perp.shape)

    def full(self):
        S = np.zeros(full_shape([self._d.shape, self._Q.shape]) + (self.dim, self.dim))
        for idx in itertools.product(*map(range, S.shape[:-2])):
            d_idx, Q_idx = broadcast(idx, self._d.shape), broadcast(idx, self._Q.shape)
            d, Q = self._d[d_idx], self._Q[Q_idx]
            S[idx + (Ellipsis,)] = np.dot(Q*d, Q.T)

            sp_idx = broadcast(idx, self._s_perp.shape)
            sp = self._s_perp[sp_idx]
            S[idx + (Ellipsis,)] += (np.eye(self.dim) - np.dot(Q, Q.T)) * sp
        return FullMatrix(S)

    def copy(self):
        return EigMatrix(self._d.copy(), self._Q.copy(), self._s_perp.copy(), self.dim)

    def elt(self, i, j):
        # TODO: make this efficient
        return self.col(j)[..., i]

    def col(self, j):
        # TODO: make this efficient
        x = np.zeros((1,) * self.ndim + (self.dim,))
        x[..., j] = 1
        return self.dot(x)

    def __slice__(self, slc):
        d_slc = process_slice(slc, self._d.shape, 0)
        Q_slc = process_slice(slc, self._Q.shape, 0)
        sp_slc = process_slice(slc, self._s_perp.shape, 0)

        if all([type(s) == int for s in slc]):
            # NumPy doesn't like zero-dimensional object arrays
            return FixedEigMatrix(self._d[d_slc], self._Q[Q_slc], self._s_perp[sp_slc])
        else:
            return EigMatrix(self._d[d_slc], self._Q[Q_slc], self._s_perp[sp_slc], self.dim)

    def __setslice__(self, slc, other):
        raise NotImplementedError()

    def dot(self, x):
        result = _QDQ_x(self._Q, self._d, x)
        x_perp = x - _QDQ_x(self._Q, self._d**0., x)
        result += x_perp * self._s_perp[..., nax]
        return result

    def qform(self, x):
        result = _x_QDQ_x(self._Q, self._d, x)
        x_perp = x - _QDQ_x(self._Q, self._d**0., x)
        result += (x_perp ** 2).sum(-1) * self._s_perp
        return result

    def pinv(self):
        new_s_perp = np.where(self._s_perp > 0., 1. / self._s_perp, 0.)
        return EigMatrix(1. / self._d, self._Q, new_s_perp, self.dim)

    def inv(self):
        return EigMatrix(1. / self._d, self._Q, 1. / self._s_perp, self.dim)

    def __add__(self, other):
        if isinstance(other, EyeMatrix):
            return EigMatrix(self._d + other._s, self._Q, self._s_perp + other._s, self.dim)
        else:
            return self.full() + other

    def __sub__(self, other):
        return self + other * -1

    def __mul__(self, other):
        return EigMatrix(other * self._d, self._Q, other * self._s_perp, self.dim)

    def sum(self, axis):
        return self.full().sum(axis)

    def logdet(self):
        d, s = self._d, self._s_perp
        fsh = full_shape([d.shape, s.shape])
        logdet = np.zeros(fsh)
        for idx in itertools.product(*map(range, fsh)):
            d_idx, s_idx = broadcast(idx, d.shape), broadcast(idx, s.shape)
            logdet[idx] = np.log(d[d_idx]).sum() + \
                          (self.dim - d[d_idx].size) * np.log(s[s_idx])
        return logdet

    def alat(self, A):
        return self.full().alat(A)

    def rescale(self, a):
        a = np.array(a)
        return EigMatrix(a**2 * self._d, self._Q, a**2 * self._s_perp, self.dim)

    def conv(self, other):
        if isinstance(other, EyeMatrix):
            s_perp_new = 1. / (1. / self._s_perp + 1. / other._s)
            d_new = 1. / (1. / self._d + 1. / other._s)
            return EigMatrix(d_new, self._Q, s_perp_new, self.dim)
        else:
            return self.full().conv(other)

    def sqrt_dot(self, x):
        ans_proj = _QDQ_x(self._Q, self._d ** 0.5, x)
        perp = x - _QDQ_x(self._Q, self._d**0., x)
        ans_perp = np.sqrt(self._s_perp) * perp
        return ans_proj + ans_perp

    def add_dummy_dimension(self):
        return self.full().add_dummy_dimension()

    def to_eig(self):
        return self

    @staticmethod
    def random(d_shape, Q_shape, sp_shape, dim, low_rank=False):
        s_perp = np.random.gamma(1., 1., size=sp_shape)

        smsh = tuple(np.array([d_shape, Q_shape]).min(0))
        if low_rank:
            rank = np.random.randint(1, dim+1, size=smsh)
        else:
            rank = dim * np.ones(smsh, dtype=int)

        d = np.zeros(d_shape, dtype=object)
        for idx in itertools.product(*map(range, d_shape)):
            sm_idx = broadcast(idx, smsh)
            d[idx] = np.random.gamma(1., 1., size=rank[sm_idx])

        Q = np.zeros(Q_shape, dtype=object)
        for idx in itertools.product(*map(range, Q_shape)):
            sm_idx = broadcast(idx, smsh)
            Q[idx], _ = np.linalg.qr(np.random.normal(size=(dim, rank[sm_idx])))

        return EigMatrix(d, Q, s_perp, dim)


class FixedEigMatrix(BaseMatrix):
    def __init__(self, d, Q, s_perp):
        BaseMatrix.__init__(self)
        d, Q, s_perp = match_shapes([('d', d, 1), ('Q', Q, 2), ('s_perp', s_perp, 0)])
        self._d = d
        self._Q = Q
        self.dim = Q.shape[-2]
        self.rank = Q.shape[-1]
        self.ndim = Q.ndim - 2
        self.shape = full_shape([d.shape[:-1], Q.shape[:-2], s_perp.shape])
        self._s_perp = s_perp
        self.shape_str = 'd=%s Q=%s s_perp=%s' % (d.shape, Q.shape, s_perp.shape)

    def full(self):
        S = dot(self._Q, self._d[..., nax] * transp(self._Q))
        S += (np.eye(self.dim) - dot(self._Q, transp(self._Q))) * self._s_perp[..., nax, nax]
        return FullMatrix(S)

    def copy(self):
        return FixedEigMatrix(self._d.copy(), self._Q.copy(), self._s_perp.copy())

    def elt(self, i, j):
        # TODO: make this efficient
        return self.col(j)[..., i]

    def col(self, j):
        # TODO: make this efficient
        x = np.zeros((1,) * self.ndim + (self.dim,))
        x[..., j] = 1
        return self.dot(x)

    def __slice__(self, slc):
        d_slc = process_slice(slc, self._d.shape, 1)
        Q_slc = process_slice(slc, self._Q.shape, 2)
        sp_slc = process_slice(slc, self._s_perp.shape, 0)
        return FixedEigMatrix(self._d[d_slc], self._Q[Q_slc], self._s_perp[sp_slc])

    def __setslice__(self, slc, other):
        raise NotImplementedError()

    def dot(self, x):
        result = dot(self._Q, self._d * dot(transp(self._Q), x))
        x_perp = x - dot(self._Q, dot(transp(self._Q), x))
        result += x_perp * self._s_perp[..., nax]
        return result

    def qform(self, x):
        result = (self._d * dot(transp(self._Q), x) ** 2).sum(-1)
        x_perp = x - dot(self._Q, dot(transp(self._Q), x))
        result += (x_perp ** 2).sum(-1) * self._s_perp
        return result

    def pinv(self):
        new_s_perp = np.where(self._s_perp > 0., 1. / self._s_perp, 0.)
        return FixedEigMatrix(1. / self._d, self._Q, new_s_perp)

    def inv(self):
        return FixedEigMatrix(1. / self._d, self._Q, 1. / self._s_perp)

    def __add__(self, other):
        if isinstance(other, EyeMatrix):
            return FixedEigMatrix(self._d + other._s[..., nax], self._Q, self._s_perp + other._s)
        else:
            return self.full() + other

    def __sub__(self, other):
        return self + other * -1

    def __mul__(self, other):
        return FixedEigMatrix(other[..., nax] * self._d, self._Q, other * self._s_perp)

    def sum(self, axis):
        return self.full().sum(axis)

    def logdet(self):
        return np.log(self._d).sum(-1) + (self.dim - self.rank) * np.log(self._s_perp)

    def alat(self, A):
        return self.full().alat(A)

    def rescale(self, a):
        a = np.array(a)
        return FixedEigMatrix(a[..., nax]**2 * self._d, self._Q, a**2 * self._s_perp)

    def conv(self, other):
        if isinstance(other, EyeMatrix):
            s_perp_new = 1. / (1. / self._s_perp + 1. / other._s)
            d_new = 1. / (1. / self._d + 1. / other._s[..., nax])
            return FixedEigMatrix(d_new, self._Q, s_perp_new)
        else:
            return self.full().conv(other)

    def sqrt_dot(self, x):
        result = dot(self._Q, np.sqrt(self._d) * dot(transp(self._Q), x))
        x_perp = x - dot(self._Q, dot(transp(self._Q), x))
        result += x_perp * np.sqrt(self._s_perp[..., nax])
        return result

    def add_dummy_dimension(self):
        return self.full().add_dummy_dimension()

    def to_eig(self):
        return self


    @staticmethod
    def random(d_shape, Q_shape, sp_shape, dim, rank=None):
        if rank is None:
            rank = dim
        ndim = len(d_shape)
        temp = np.random.normal(size=Q_shape + (dim, rank))
        Q, _ = array_map(np.linalg.qr, [temp], ndim)
        d = np.random.gamma(1., 1., size=d_shape + (rank,))
        sp = np.random.gamma(1., 1., size=sp_shape)
        return FixedEigMatrix(d, Q, sp)


def proj_psd(H):
    '''
    Makes stuff psd I presume? Comments welcome.
    '''
    assert np.allclose(H, H.T), 'not symmetric'
    d, Q = scipy.linalg.eigh(H)
    d = np.clip(d, 1e-8, np.infty)
    return np.dot(Q, d[:, nax] * Q.T)
    
def laplace_approx(nll, opt_hyper, hessian, prior_var=100):
    #### FIXME - Believed to have a bug
    ####       - Might be MATLAB though - test this code on some known integrals
    d = opt_hyper.size
    
    hessian = proj_psd(hessian)

    # quadratic centered at opt_hyper with maximum -nll
    evidence = gaussians.Potential(np.zeros(d), FullMatrix(hessian), -nll)
    evidence = evidence.translate(opt_hyper)

    # zero-centered Gaussian
    prior = gaussians.Potential.from_moments_iso(np.zeros(d), prior_var)

    # multiply the two Gaussians and integrate the result
    return -(evidence + prior).integral()
