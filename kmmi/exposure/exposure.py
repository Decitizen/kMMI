# Utility functions
import numpy as np
from numba import *
import networkx as nx

flatten = lambda x : list(chain.from_iterable(x))
overlap_coefficient = lambda A,B: len(A & B) / np.min([len(A),len(B)])

@njit
def prune_by_edge_weight(E: np.array, threshold: float=1e-3):
    """
    Prune edges for which weight is below given threshold value. 
    """
    n = E.shape[0]
    Et = np.zeros((n,n)) 
    for i in range(n):
        for j in range(n):
            if Et[i,j] < threshold:
                Et[i,j] = 0.0
            else:
                Et[i,j] = E[i,j]
    return Et

@njit
def compute_btw_exposure_matrix(A: np.array, a: float, l1: float=None):
    """Compute BTW exposure pathway matrix given 
       A (numpy array): adjacency matrix of dimensions n x n 
                        (positive semi-definite, directed, weighted)
       a (float): alpha coefficient
       l1 (float): largest (positive) eigenvalue of the A matrix
    """
    if l1 is not None:
        assert a < 1 / l1, 'a (alpha) should be smaller than the ' \
                           'reciprocal of the largest eigenvalue of A'
    n = A.shape[0]
    I = np.identity(n)
    E = np.linalg.inv(I - a*A)
    assert np.all(E.ravel() >= 0.0), 'There are negative weights in the E'
    
    return E

@njit
def compute_nbtw_exposure_matrix(A: np.array, a: float, weighted=True, lc1: float=None):
    """Compute NBTW exposure matrix
       A (numpy array): adjacency matrix
       a (float): alpha coefficient
       weighted (bool): is A weighted (if True, lc1 is ignored since limit of convergence is not known)
       lc1 (float): spectral radius of matrix C, can be calculated by get_spectral_radius_C.
                    If no directed cycles, use None.
                    Ignored when weighted == True
    """
    assert A.shape[0] == A.shape[1], 'Matrix is not symmetric'
    if weighted:
        n = A.shape[0]
        I = np.identity(n)
        S = np.multiply(A,np.transpose(A)) # elementwise multiplication
        R = np.sqrt(S)
        helper1 = I + np.diag(np.diag(np.divide(a*R,1-a*R) @ np.divide(a*R,1+a*R)))
        helper2 = np.divide(a*A,1-(a**2)*S)
        E = np.linalg.inv(helper1-helper2)
    else:
        if lc1 is not None:
            assert a < 1.0 / lc1, 'a should be smaller than 1 / spectral radius of C'
        n = A.shape[0]
        I = np.identity(n)
        D = np.diag(np.diag(A @ A))
        S = np.zeros(A.shape)
        for ii in range(A.shape[0]):
            for jj in range(ii,A.shape[0]):
                e = A[ii,jj]*A[jj,ii]
                S[ii,jj] = e
                S[jj,ii] = e
        M = I - A*a + (D-I)*(a**2) + (A-S)*(a**3)
        E = (1-a**2)*np.linalg.inv(M)
    assert np.all(E.ravel() >= 0.0), 'There are negative weights in the E'
    
    return E

@njit
def get_spectral_radius_A(A: np.array, iters: int=10, seed: int=None):
    """Get spectral radius of matrix A by power iteration
    """
    assert A.shape[0] == A.shape[1], 'Matrix is not symmetric'
    if seed is not None:
        np.random.seed(seed)
    n = A.shape[0]
    
    b_k = np.random.rand(n).astype(np.float64)
    b_k = b_k/np.sqrt((b_k**2).sum())
    b_k_next = np.zeros(n).astype(np.float64)
    
    for ii in range(iters):
        b_k_next = A @ b_k
        b_k_next = b_k_next/np.sqrt((b_k_next**2).sum())
        b_k = b_k_next
        
    # Rayleigh quotient
    b_k_A = A @ b_k
    l = (b_k.transpose() @ b_k_A) / (b_k.transpose() @ b_k)
    
    return np.abs(l)

@njit
def get_spectral_radius_C(A: np.array, iters=10, seed=None):
    """Get spectral radius of matrix C (derived from A) by power iteration
    """
    assert A.shape[0] == A.shape[1], 'Matrix is not symmetric'
    if seed is not None:
        np.random.seed(seed)
    n = A.shape[0]
    I = np.identity(n).astype(np.float64)
    D = np.diag(np.diag(A @ A)).astype(np.float64)
    S = np.zeros(A.shape).astype(np.float64)
    
    for ii in range(A.shape[0]):
        for jj in range(ii,A.shape[0]):
            e = A[ii,jj]*A[jj,ii]
            S[ii,jj] = e
            S[jj,ii] = e
    
    helper_1 = I-D
    helper_2 = S-A
    b_k = np.random.rand(n*3).astype(np.float64)
    b_k = b_k/np.sqrt((b_k**2).sum())
    b_k_next = np.zeros(n*3).astype(np.float64)
    
    for ii in range(iters):
        b_k_next[0:n] = A @ b_k[0:n] + helper_1 @ b_k[n:n*2] + helper_2 @ b_k[n*2:n*3]
        b_k_next[n:n*2] = I @ b_k[0:n]
        b_k_next[n*2:n*3] = I @ b_k[n:n*2]
        b_k_next = b_k_next/np.sqrt((b_k_next**2).sum())
        b_k = b_k_next
    
    # Rayleigh quotient
    b_k_C = np.zeros(n*3).astype(np.float64)
    b_k_C[0:n] = A @ b_k[0:n] + helper_1 @ b_k[n:n*2] + helper_2 @ b_k[n*2:n*3]
    b_k_C[n:n*2] = I @ b_k[0:n]
    b_k_C[n*2:n*3] = I @ b_k[n:n*2]
    l = (b_k.transpose() @ b_k_C) / (b_k.transpose() @ b_k)
    
    return np.abs(l)

def binary_search_spectral_radius(katz_l1, A, weighted=True, verbose=False, n_iters=None):
    """Compute approximate spectral radius for the weighted directed nbtw 
    adjacency matrix C. This implementation uses binary search to search 
    for the limit where the C diverges.
    """
    if n_iters is None:
        n_iters = int(np.log10(A.shape[0])**-1 * 30) 
    alpha_max = alpha = katz_l1**-1
    # Search for the upper bound for the search range
    while True:
        try:
            Ed = compute_nbtw_exposure_matrix(A, alpha, weighted=weighted)
            if verbose: print(':: alpha {:.4f} converged successfully'.format(alpha_max))
            alpha_max = alpha
            alpha *= 2
        except Exception as e:
            if verbose: print(':: Initial upper bound for the search range found: {:.4f}'.format(alpha))
            if verbose: print(':: Radius of convergence is between: [{:.4f},{:.4f}]'.format(alpha**-1, alpha_max**-1))
            break
    L = alpha_max
    R = alpha_max * 2
    # Binary search
    for i in range(n_iters):
        m = (L + R) / 2
        try:
            Ed = compute_nbtw_exposure_matrix(A, m, weighted=weighted)
            if verbose: print(':: alpha {:.4f} converged successfully'.format(m))
            alpha_max = L = m
        except:                    
            R = m
            if verbose: print(':: {:.4f} not valid upper bound'.format(m))
            if verbose: print(':: Radius of convergence is between: [{:.4f},{:.4f}]'.format(m**-1, (alpha_max)**-1))
    return L**-1

def to_exposure_matrix(G: nx.DiGraph, f: float, w_threshold = 0.001, nbtw=True, verbose=False):
    """Transform network into exposure matrix, use nbtw attribute to control 
    non-backtracking behavior.
    """
    A = nx.to_numpy_array(G, dtype=np.float64)
    
    if verbose: print(':: Pruning light edges (< {}) to reduce the density of the network.'.format(w_threshold))
    A = prune_by_edge_weight(A, threshold=w_threshold)
    
    ##  2.2. Compute exposure matrix
    if verbose: print(':: Computing spectral radius of A')
    l1 = get_spectral_radius_A(A, iters=20)
    
    if nbtw:
        if verbose: print(':: Computing spectral radius of C')        
        l1_nbtw = binary_search_spectral_radius(l1, A, verbose=verbose)
        alpha = l1_nbtw**-1 * f
        if verbose: print(':: Alpha of NBTW: {:.2f}'.format(alpha))
        if verbose: print(':: Computing exposure matrix.')
        
        E_directed = compute_nbtw_exposure_matrix(A, alpha, weighted=True)
        if verbose: print(E_directed.round(4))
        
    else: 
        if verbose: print(':: PF eigenvalue for BTW: {:.2f}'.format(l1))
        alpha = l1**-1 * f
        if verbose: print(':: Alpha of BTW: {:.2f}'.format(alpha))
        if verbose: print(':: Computing exposure matrix.')
        E_directed = compute_btw_exposure_matrix(A, alpha, l1)
        
    return E_directed

def transform_to_hks_input(E):
    """Transforms provided numpy array with appropriate scaling."""
    # Scaling with max and adding positive constant value ensures positive weights 
    epsilon = 1e-4
    Emax = np.max(E)
    E_input = -E+Emax+epsilon
    return E_input

def rescale_positive(G, w_min, norm=True):
    """Rescales edge weights to retain positive weights."""
    assert w_min < 0, 'Minimum value is non-negative, no need to rescale weights.'
    
    epsilon = 1e-4
    A = nx.to_numpy_array(G)
    A += (-1*w_min + epsilon)
    
    return nx.Graph(A / A.max())

@njit
def coarse_grain_exposure_matrix(Vs_hat, E_input, use_mutual_exposure=True, 
                                 penalty_weight=4, verbose=False):
    """
    Merge together seed exposures over matrix E for coarse grained
    node set Vs_hat. Output matrix E_hat of dimensions p x p, where
    p is the number of nodes in set Vs_hat. If use_mutual_exposure is
    true, the matrix EE^T will be used for the exposure computations.
    """
    
    E = E_input @ E_input.T if use_mutual_exposure else E_input
    p = Vs_hat.shape[0]
    T = np.zeros((2,p,p))
    E = E / E.max()
    
    for vi in range(p):
        for vj in range(p):
            Vi = Vs_hat[vi,:]
            Vj = Vs_hat[vj,:]
            for i in Vi:
                if i < 0: break
                for j in Vj:
                    if j < 0: break
                    if vi != vj:
                        T[0,vi,vj] += E[i,j]
                        if i == j:
                            T[1,vi,vj] += 1
    
    E_hat = (T[0,:,:] / T[0,:,:].max()) + (T[1,:,:] * penalty_weight)
    
    return E_hat / E_hat.max()