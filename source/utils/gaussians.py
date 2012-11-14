import numpy as np
nax = np.newaxis

import psd_matrices

#from profiler import profiled
import profiler
profiled = profiler.profiled('gaussians')
from misc import _err_string, process_slice, my_sum, match_shapes, dot, full_shape, broadcast, set_err_info, transp


class Potential():
    def __init__(self, J, Lambda, Z):
        J, Lambda, Z = match_shapes([('J', J, 1), ('Lambda', Lambda, 0), ('Z', Z, 0)])
        self._J = J
        self._Lambda = Lambda
        self._Z = Z
        self.shape = full_shape([J.shape[:-1], Lambda.shape, Z.shape])
        self.ndim = J.ndim - 1
        self.dim = J.shape[-1]
        self.shape_str = '%s J=%s Z=%s %s' % (Lambda.__class__, J.shape, Z.shape, Lambda.shape_str)
        self.mutable = False

    def set_mutable(self, m):
        # copy everything, just in case
        self._J = self._J.copy()
        self._Lambda = self._Lambda.copy()
        self._Z = self._Z.copy()

        self.mutable = m
        self._Lambda.set_mutable(m)
        

    @profiled
    def full(self):
        return Potential(self._J, self._Lambda.full(), self._Z)

    @profiled
    def copy(self):
        return Potential(self._J.copy(), self._Lambda.copy(), self._Z.copy())

    @profiled
    def score(self, x):
        return -0.5 * self._Lambda.qform(x) + (self._J * x).sum(-1) + self._Z

    @profiled
    def loglik(self, x):
        return self.score(x)

    @profiled
    def flip(self):
        return Potential(-self._J, self._Lambda, self._Z)

    @profiled
    def translate(self, dmu):
        new_J = self._J + self._Lambda.dot(dmu)
        linv = self._Lambda.pinv()
        new_Z = self._Z + 0.5 * linv.qform(self._J) - 0.5 * linv.qform(new_J)
        return Potential(new_J, self._Lambda, new_Z)

    def __getitem__(self, slc):
        return self.__slice__(slc)

    @profiled
    def __slice__(self, slc):
        J_slc = process_slice(slc, self._J.shape, 1)
        Lambda_slc = process_slice(slc, self._Lambda.shape, 0)
        Z_slc = process_slice(slc, self._Z.shape, 0)
        return Potential(self._J[J_slc], self._Lambda[Lambda_slc], self._Z[Z_slc])

    def __setitem__(self, slc, other):
        return self.__setslice__(slc, other)

    @profiled
    def __setslice__(self, slc, other):
        if not self.mutable:
            raise RuntimeError('Attempt to modify immutable potential')
        J_slc = process_slice(slc, self._J.shape, 1)
        Lambda_slc = process_slice(slc, self._Lambda.shape, 0)
        Z_slc = process_slice(slc, self._Z.shape, 0)
        self._J[J_slc] = other._J
        self._Lambda[Lambda_slc] = other._Lambda
        self._Z[Z_slc] = other._Z

    @profiled
    def __add__(self, other):
        return Potential(self._J + other._J, self._Lambda + other._Lambda, self._Z + other._Z)

    @profiled
    def __sub__(self, other):
        return Potential(self._J - other._J, self._Lambda - other._Lambda, self._Z - other._Z)

    @profiled
    def __mul__(self, other):
        other = np.asarray(other)
        return Potential(self._J * other[..., nax], self._Lambda * other, self._Z * other)

    @profiled
    def __rmul__(self, other):
        return self * other

    @profiled
    def sum(self, axis):
        assert type(axis) == int and 0 <= axis < self.ndim
        return Potential(my_sum(self._J, axis, self.shape[axis]),
                                my_sum(self._Lambda, axis, self.shape[axis]),
                                my_sum(self._Z, axis, self.shape[axis]))

    @profiled
    def conv(self, other):
        J1, J2, Lambda1, Lambda2, Z1, Z2 = self._J, other._J, self._Lambda, other._Lambda, self._Z, other._Z
        LL = Lambda1 + Lambda2
        P = LL.pinv()
        Lambda_c = Lambda1.conv(Lambda2)
        J_c = Lambda1.dot(P.dot(J2)) + Lambda2.dot(P.dot(J1))
        Z_c = 0.5 * P.qform(J1 - J2) + 0.5 * self.dim * np.log(2*np.pi) - 0.5 * LL.logdet() + Z1 + Z2
        return Potential(J_c, Lambda_c, Z_c)

    @profiled
    def transform(self, A):
        J = dot(transp(A), self._J)
        Lambda = self._Lambda.alat(transp(A))
        return Potential(J, Lambda, self._Z)

    @profiled
    def rescale(self, a):
        a = np.array(a)
        J = a[..., nax] * self._J
        Lambda = self._Lambda.rescale(a)
        return Potential(J, Lambda, self._Z)



    @profiled
    def integral(self):
        J, Lambda, Z = self._J, self._Lambda, self._Z
        linv = Lambda.pinv()
        return 0.5 * self.dim * np.log(2*np.pi) - 0.5 * Lambda.logdet() + 0.5 * linv.qform(J) + Z

    @profiled
    def renorm(self):
        return Potential(self._J, self._Lambda, self._Z - self.integral())

    @profiled
    def add_dummy_dimension(self):
        J = np.zeros(self._J.shape[:-1] + (self.dim + 1,))
        J[..., 1:] = self._J
        Lambda = self._Lambda.add_dummy_dimension()
        return Potential(J, Lambda, self._Z)

    @profiled
    def to_eig(self):
        return Potential(self._J, self._Lambda.to_eig(), self._Z)

    @staticmethod
    @profiled
    def from_moments(mu, Sigma):
        return Distribution(mu, Sigma).to_potential()

    @staticmethod
    @profiled
    def from_moments_full(mu, Sigma):
        return Distribution(mu, psd_matrices.FullMatrix(Sigma)).to_potential()

    @staticmethod
    @profiled
    def from_moments_diag(mu, sigma_sq):
        return Distribution(mu, psd_matrices.DiagonalMatrix(sigma_sq)).to_potential()

    @staticmethod
    @profiled
    def from_moments_iso(mu, sigma_sq):
        sigma_sq = np.asarray(sigma_sq)
        return Distribution(mu, psd_matrices.EyeMatrix(sigma_sq, mu.shape[-1])).to_potential()

    @profiled
    def allclose(self, other):
        J_err = _err_string(self._J, other._J)
        Lambda_err = _err_string(self._Lambda.full()._S, other._Lambda.full()._S)
        Z_err = _err_string(self._Z, other._Z)
        set_err_info('gaussians', [('J', J_err), ('Lambda', Lambda_err), ('Z', Z_err)])

        return np.allclose(self._J, other._J) and \
               self._Lambda.allclose(other._Lambda) and \
               np.allclose(self._Z, other._Z)

    @profiled
    def to_distribution(self):
        Sigma = self._Lambda.inv()
        mu = Sigma.dot(self._J)
        Z = self._Z + 0.5 * self.dim * np.log(2*np.pi) + 0.5 * Sigma.logdet() + 0.5 * Sigma.qform(self._J)
        return Distribution(mu, Sigma, Z)

    @staticmethod
    def random(J_shape, Z_shape, Lambda, dim):
        J = np.random.normal(size=J_shape + (dim,))
        Z = np.random.normal(size=Z_shape)
        return Potential(J, Lambda, Z)

    @profiled
    def conditionals(self, X):
        return Conditionals.from_potential(self, X)

    @profiled
    def mu(self):
        return self._Lambda.pinv().dot(self._J)

    
class Distribution:
    def __init__(self, mu, Sigma, Z=0.):
        mu, Sigma, Z = match_shapes([('mu', mu, 1), ('Sigma', Sigma, 0), ('Z', Z, 0)])
        self._mu = mu
        self._Sigma = Sigma
        self._Z = Z
        self.dim = mu.shape[-1]
        self.ndim = mu.ndim - 1
        self.shape = full_shape([Sigma.shape, mu.shape[:-1], Z.shape])
        self.shape_str = '%s mu=%s Z=%s %s' % (Sigma.__class__, mu.shape, Z.shape, Sigma.shape_str)

    def allclose(self, other):
        return np.allclose(self._mu, other._mu) and \
               np.allclose(self._Z, other._Z) and \
               self._Sigma.allclose(other._Sigma)

    @profiled
    def full(self):
        return Distribution(self._mu, self._Sigma.full(), self._Z)

    @profiled
    def __add__(self, other):
        return Distribution(self._mu + other._mu, self._Sigma + other._Sigma, self._Z + other._Z)

    @profiled
    def translate(self, dmu):
        return Distribution(self._mu + dmu, self._Sigma, self._Z)

    @profiled
    def to_potential(self):
        Lambda = self._Sigma.inv()
        J = Lambda.dot(self._mu)
        Z = -0.5 * self.dim * np.log(2*np.pi) - 0.5 * self._Sigma.logdet() - 0.5 * self._Sigma.qform(J) + self._Z
        return Potential(J, Lambda, Z)

    @profiled
    def sample(self):
        return self._mu + self._Sigma.sqrt_dot(np.random.normal(size=self.shape + (self.dim,)))

    @profiled
    def transform(self, A):
        return Distribution(dot(A, self._mu), self._Sigma.alat(A), self._Z)

    @profiled
    def __slice__(self, slc):
        mu_slc = process_slice(slc, self._mu.shape, 1)
        Sigma_slc = process_slice(slc, self._Sigma.shape, 0)
        Z_slc = process_slice(slc, self._Z.shape, 0)
        return Distribution(self._mu[mu_slc], self._Sigma[Sigma_slc], self._Z[Z_slc])

    @profiled
    def loglik(self, x):
        return self.to_potential().score(x)

    @staticmethod
    def from_moments_full(mu, Sigma, Z=0.):
        return Distribution(mu, psd_matrices.FullMatrix(Sigma), Z)

    @staticmethod
    def from_moments_diag(mu, sigma_sq, Z=0.):
        return Distribution(mu, psd_matrices.FullMatrix(np.diag(sigma_sq)), Z)

    @staticmethod
    def from_moments_iso(mu, sigma_sq, Z=0.):
        dim = mu.shape[-1]
        return Distribution(mu, psd_matrices.EyeMatrix(sigma_sq, dim), Z)

    def mu(self):
        return self._mu

    def Sigma(self):
        return self._Sigma.full()._S

    def Z(self):
        return self._Z


class Conditionals:
    def __init__(self, Lambda, J_diff, Z_diff, X):
        Lambda, J_diff, X = match_shapes([('Lambda', Lambda, 0), ('J_diff', J_diff, 1), ('X', X, 1)])
        self._Lambda = Lambda
        self._J_diff = J_diff.copy()
        self._Z_diff = Z_diff.copy()
        self._X = X.copy()
        self.dim = self._J_diff.shape[-1]
        self.ndim = self._J_diff.ndim - 1
        self.shape = full_shape([Lambda.shape, J_diff.shape[:-1], X.shape[:-1]])
        self.shape_str = '%s J_diff=%s X=%s %s' % (Lambda.__class__, J_diff.shape, X.shape, Lambda.shape_str)

        ## can't have EigMatrix of zero dimensions, since NumPy doesn't like zero-dimensional object arrays
        #if self.shape == () and isinstance(Lambda, psd_matrices.EigMatrix):
        #    self._Lambda = self._Lambda.full()

    def allclose(self, other):
        return self._Lambda.allclose(other._Lambda) and \
               np.allclose(self._J_diff, other._J_diff) and \
               np.allclose(self._Z_diff, other._Z_diff) and \
               np.allclose(self._X, other._X)
        
    @profiled
    def __slice__(self, slc):
        Lambda_slc = process_slice(slc, self._Lambda.shape, 0)
        J_slc = process_slice(slc, self._J_diff.shape, 1)
        Z_slc = process_slice(slc, self._Z_diff.shape, 0)
        X_slc = process_slice(slc, self._X.shape, 1)
        return Conditionals(self._Lambda[Lambda_slc], self._J_diff[J_slc], self._Z_diff[Z_slc], self._X[X_slc])

    @profiled
    def conditional_for(self, i):
        Lambda = psd_matrices.EyeMatrix(self._Lambda.elt(i, i), 1)
        return Potential(self._J_diff[..., i:i+1].copy(), Lambda, self._Z_diff).translate(self._X[..., i:i+1])

    @profiled
    def assign(self, j, x_new):
        diff = x_new - self._X[..., j]
        self._X[..., j] = x_new
        self._Z_diff += self._J_diff[..., j] * diff + \
                        -0.5 * self._Lambda.elt(j, j) * diff ** 2
        self._J_diff -= diff[..., nax] * self._Lambda.col(j)


    @profiled
    def assign_one(self, idx, j, x_new):
        if type(idx) == int:
            idx = (idx,)
        diff = x_new - self._X[idx + (j,)]
        self._X[idx + (j,)] = x_new
        Lambda_idx = broadcast(idx, self._Lambda.shape)
        self._Z_diff[idx] += self._J_diff[idx + (j,)] * diff + \
                             -0.5 * self._Lambda[Lambda_idx].elt(j, j) * diff ** 2
        self._J_diff[idx + (slice(None),)] -= diff * self._Lambda[Lambda_idx].col(j)

    @staticmethod
    @profiled
    def from_potential(pot, X):
        J_diff = pot._J - pot._Lambda.dot(X)
        Z_diff = pot.score(X)
        return Conditionals(pot._Lambda, J_diff, Z_diff, X)


