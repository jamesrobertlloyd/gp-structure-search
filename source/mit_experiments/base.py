import cPickle
import numpy as np
nax = np.newaxis
import os
import scipy.linalg

import datasets
import flexiblekernel
import grammar
import mit_job_controller

def find_duplicates(K_list, rel_cutoff=0.01):
    """Given a list of kernel matrices sorted in order from best to worst, remove
    all duplicates, where two matrices are duplicates if the Euclidean distance
    between them is smaller than a cutoff. The cutoff is given by rel_cutoff times
    the mean distance. The kernel matrices are flattened
    into vectors, so some other approximation (such as a random projection) could
    be used instead. Returns a boolean vector where the True entries denote duplicates."""
    nker = len(K_list)
    K_vec = np.array([K.ravel() for K in K_list])

    # if this is triggered, need to compute dist in a memory efficient way
    assert K_vec.size * nker < 1e8

    dist = np.sqrt(np.sum((K_vec[:, nax, :] - K_vec[nax, :, :]) ** 2, axis=2))

    cutoff = rel_cutoff * dist.mean()

    isdup = np.zeros(nker, dtype=bool)
    for i in range(nker):
        if isdup[i]:
            continue
        for j in range(i + 1, nker):
            if dist[i, j] < cutoff:
                isdup[j] = True

    return isdup
    
def remove_duplicates(kernels, X, num_subsample=None, proj_dim=None, rel_cutoff=0.01):
    """Remove the duplicate kernels, i.e. remove the worse of any pair for which the
    distance is smaller than rel_cutoff times the average distance. Optionally perform
    two approximations to save time and memory:
        - if num_subsample is not None, choose num_subsample data points at random
        - if proj_dim is not None, project onto a random subspace of dimension proj_dim"""
    # sort from best to worst
    kernels = sorted(kernels, key=flexiblekernel.ScoredKernel.score)

    # approximation: choose a random subset of the data
    if num_subsample is not None and num_subsample < X.shape[0]:
        idxs = np.random.permutation(X.shape[0])[:num_subsample]
        X = X[idxs, :]
    ndata = X.shape[0]

    # approximation: project onto random subspace
    if proj_dim is not None and proj_dim < ndata**2:
        A = np.random.normal(size=(ndata**2, proj_dim))
        Q, _ = scipy.linalg.qr(A, mode='economic')
    else:
        Q = None

    K_list = mit_job_controller.compute_K(kernels, X, Q)

    isdup = find_duplicates(K_list, rel_cutoff)
    idxs = np.where(-isdup)[0]
    return [kernels[i] for i in idxs]
    
def remove_nan_scored_kernels(scored_kernels):    
    return [k for k in scored_kernels if not np.isnan(k.score())]

class SearchParams:
    def __init__(self, n_restarts, restart_std, num_subsample, proj_dim, rel_cutoff,
                 num_winners, num_expand):
        self.n_restarts = n_restarts
        self.restart_std = restart_std
        self.num_subsample = num_subsample
        self.proj_dim = proj_dim
        self.rel_cutoff = rel_cutoff
        self.num_winners = num_winners
        self.num_expand = num_expand

    @staticmethod
    def default():
        return SearchParams(n_restarts=1, restart_std=2., num_subsample=300, proj_dim=1000,
                            rel_cutoff=0.01, num_winners=250, num_expand=2)

def perform_search(X, y, scheduler, max_depth, params, verbose=False, output_fname_fn=None):
    D = X.shape[1]
    current_kernels = list(flexiblekernel.base_kernels(D))

    all_scored_kernels = []
    scored_kernels_by_level = []
    for depth in range(max_depth):
        if verbose:
            print 'Level', depth + 1
        current_kernels = flexiblekernel.add_random_restarts(current_kernels, params.n_restarts,
                                                             params.restart_std)

        if verbose:
            print 'Evaluating kernels...'
        scored_kernels = scheduler.evaluate_kernels(current_kernels, X, y)

        scored_kernels = remove_nan_scored_kernels(scored_kernels)
        scored_kernels.sort(key=flexiblekernel.ScoredKernel.score)
        scored_kernels = scored_kernels[:params.num_winners]
        if verbose:
            print 'Removing duplicates...'
        scored_kernels = remove_duplicates(scored_kernels, X, params.num_subsample, params.proj_dim,
                                           params.rel_cutoff)
        scored_kernels.sort(key=flexiblekernel.ScoredKernel.score)

        all_scored_kernels += scored_kernels
        scored_kernels_by_level.append(scored_kernels)

        best_kernels = [k.k_opt for k in scored_kernels[:params.num_expand]]
        current_kernels = grammar.expand_kernels(D, best_kernels)

        if output_fname_fn is not None:
            if verbose:
                print 'Saving results...'
            fname = output_fname_fn(depth)
            cPickle.dump(current_kernels, open(fname, 'wb'), protocol=2)

    all_scored_kernels.sort(key=flexiblekernel.ScoredKernel.score)
    return all_scored_kernels, scored_kernels_by_level


def load_data(name):
    if name == 'airline':
        X, y = datasets.airline.load_X_y()
    elif name == 'eeg-single':
        X, y = datasets.eeg.load_one_channel()
    elif name == 'eeg-all':
        X, y = datasets.eeg.load_all_channels()
    elif name == 'methane':
        X, y = datasets.methane.read_data()
    elif name == 'sea-level-monthly':
        X, y = datasets.sea_level.get_X_y('monthly')
    elif name == 'sea-level-annual':
        X, y = datasets.sea_level.get_X_y('annual')
    elif name == 'solar':
        X, y = datasets.solar.get_X_y('solar')
    else:
        fname = '../data/%s.mat' % name
        if not os.path.exists(fname):
            raise RuntimeError("Couldn't find dataset: %s" % name)
        X, y = scipy.io.loadmat(fname)

    # make sure X and y are in matrix form
    if X.ndim == 1:
        X = X[:, nax]
    if y.ndim == 1:
        y = y[:, nax]

    return X, y




