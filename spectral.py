# -*- coding: utf-8 -*-
"""Algorithms for spectral clustering"""

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
#         Brian Cheung
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause
import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state, as_float_array
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
from sklearn.cluster.k_means_ import k_means
from scipy.linalg import qr


def discretize(vectors, copy=True, max_svd_restarts=3, n_iter_max=50,
               random_state=None):
    """Search for a partition matrix (clustering) which is closest to the
    eigenvector embedding.

    Parameters
    ----------
    vectors : array-like, shape: (n_samples, n_clusters)
        The embedding space of the samples.

    copy : boolean, optional, default: True
        Whether to copy vectors, or perform in-place normalization.

    max_svd_restarts : int, optional, default: 30
        Maximum number of attempts to restart SVD if convergence fails

    n_iter_max : int, optional, default: 30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached

    random_state : int, RandomState instance or None (default)
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

    Notes
    -----

    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.

    """

    from scipy.sparse import csc_matrix, csr_matrix
    from scipy.linalg import LinAlgError

    random_state = check_random_state(random_state)

    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) \
            * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
   
    #print(np.isnan(vectors).any(), np.isinf(vectors).any())
    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]
    vectors[np.isnan(vectors)]=0
    vectors[np.isinf(vectors)]=0

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.
    while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        ##rotation = np.zeros((n_components, n_components))
        #rotation[:, 0] = vectors[random_state.randint(n_samples), :].T
        ##rotation[:, 0] = vectors.mean(axis=0)

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        ##c = np.zeros(n_samples)
        ##for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            ##c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            ##rotation[:, j] = vectors[c.argmin(), :].T
        
        rotation = np.identity(n_components)

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components)).toarray()
            
            vectors_f = vectors_discrete
            vectors_fs = np.sqrt(vectors_f.sum(axis=0))
            vectors_fs[vectors_fs==0]=1
            vectors_f = vectors_f*1.0/vectors_fs

            t_svd = vectors_f.T.dot(vectors)

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if ((abs(ncut_value - last_objective_value) < eps) or
                    (n_iter > n_iter_max)):
                has_converged = True
                #vectors_f = vectors_discrete
                #vectors_fs = np.sqrt(vectors_f.sum(axis=0))
                #vectors_f = vectors_f*1.0/vectors_fs
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError('SVD did not converge')

    return labels, vectors_f#.todense()
