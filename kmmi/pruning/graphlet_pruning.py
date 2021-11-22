import numpy as np
import networkx as nx
from numba import njit
from itertools import chain
from time import process_time
from datetime import timedelta
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from kmmi.enumeration.graphlet_enumeration import *
from kmmi.utils.utils import to_numpy_array, mean_ndiag, overlap_coefficient
from kmmi.utils.autoload import *

class Error(Exception):
    """Base class for other exceptions"""
    pass

class SeedNodeError(Error):
    """Exception raised if seed nodes are not fulfilling the desired selection criteria.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message=""):
        self.message = message + 'Increase number of seed nodes in the immediate ' \
                       'neighborhood or remove v from force-selected nodes.'
        super().__init__(self.message)

def prune_by_density(U: np.array, A: np.array, ds: np.array=None, 
                     delta_min: float=0.7, delta_max: float=1.00) -> np.array: 
    """Prune all subgraphs Gs s.t. delta_min <= density(Gs) <= delta_max"""
    if ds is None:
        _, ds = graphlet_scores(U, A)
    d_sel = (delta_min <= ds) & (ds <= delta_max) 
    d_unsel = (delta_min > ds) | (ds > delta_max)    
    assert d_sel.sum() > 0,  "All graphlets were pruned; " \
                             "selected density range may be too narrow, " \
                             "lower the delta_min or increase the delta_max " \
                             "to relax the requirement."
    return U[d_sel,:], U[d_unsel, :]

def strongly_connected(A: np.array, U: np.array):
    """Select all graphlets s in S which fulfill the definition of strongly 
    connected component in a directed network, namely that for each pair of 
    nodes u,v \in S there needs to exist at least one path for which u is 
    reachable from node v and vice versa.
    """
    SCC = lambda B: connected_components(B, directed=True, 
                        connection='strong')[1].sum() == 0
    idxs = np.apply_along_axis(lambda s: SCC(A[s,:][:,s]), 1, U)        
    return U[idxs,:]

@njit
def graphlet_scores(U: np.array, A: np.array):
    """Compute graphlet scores for graphlets in U. Graphlet scoring
    function is defined as such that, i!=j and
    $$\tau = \frac{1}{(n*(n-1))} \sum_{i,j \in s} A_{ij}$$
    where A is the weighted adjacency matrix of the subgraph G_s 
    induced by the set of nodes $s = \{u_1,u_2,...,u_n}$, and n is 
    the number of edges  in the induced subgraph G_s.
    """
    n = U.shape[0]
    ws = np.zeros(n)
    ds = np.zeros(n)
    for i in range(n):
        Ui = U[i,:]
        s = Ui[Ui >= 0]
        n_s = len(s)
        B = A[s,:][:,s]
        ws[i] = mean_ndiag(B)
        ds[i] = (B > 0.0).sum() / (n_s*(n_s-1))
        
    return ws, ds

@njit
def select_nrank(U: np.array, A: np.array, p: int, presorted=False, verbose=False):
    """Selects p highest ranking graphlets per each node in the original 
    network. Assumes that U is already ordered in the ascending ranking order.
    """
    if not presorted:
        if verbose: print(':: Computing tau scores...')
        taus, _ = graphlet_scores(U, A)
        if verbose: print(':: Sorting based on tau scores...')
        idxs = np.argsort(taus)[::-1]
        U = U[idxs]
    
    if verbose: print(f':: Selecting {p} graphlets per node...')
    n = U.shape[0]
    k = U.shape[1]
    u_sel = np.array([False]*n)
    C = np.zeros(n_v, dtype=np.int16)
    for i in range(n):
        for j in range(k):
            Uidx = U[i,j]
            if Uidx != -1:
                if C[Uidx] != p:
                    C[Uidx] += 1
                    u_sel[i] = True
    return U[u_sel,:], u_sel

def binary_search_p(U: np.array, n_v: int, tol: int=1000, n_max: int=10000, verbose=False):
    """Compute the best value of p parameter for the select_nrank function. Useful when 
    the number of graphlet candidates is larger than what the resources available for 
    running the downstream tasks that use the coarse-grained network. This implementation
    uses binary search to search for the limit such that output < n_max.
    """
    n = U.shape[0]
    n_v = A.shape[0]
    assert n > n_max, f'n: {n} > n_max: {n_max}, binary search isn\'t required'
    if n_max > 1e5: print(f'WARNING: n_max: {n_max} is higher than what the ' \
                          'pipeline has been benchmarked for.')

    if verbose:
        print(':: Initializing binary search for determining upper bound for p...')
        print(f':: Tolerance: {tol}')
    i = n_sel = 1
    while n_sel < n_max:
        p = 2**i
        _, idxs = select_nrank(U, A, p, True, verbose)
        n_sel = idxs.sum()
        i += 1

    if verbose: print(f':: Initial upper bound found for the p: {p}')
    d1 = np.inf
    S0 = idxs
    L, R = p/2, p
    while np.abs(d1) > tol:
        m = int((L + R) / 2)
        _, idxs = select_nrank(U, A, p, True, verbose)
        if idxs.sum() <= n_max:
            d1 = idxs.sum() - S0.sum()
            L = m
            if verbose: print(f':: * {m} ({idxs.sum()}) set as the lower bound')
        else:
            d1 = idxs.sum() - S0.sum()
            S0 = idxs
            R = m
            if verbose: print(f':: * {m} ({idxs.sum()}) set as the upper bound')
    if verbose: print(':: Convergence succesful, final p: {} ({} selected)'
                      .format(L, S0.sum())) 
    return L, S0

def prune_subgraphs(U: np.array, k_min: int, k_max: int):
    """For two node sets V1 and V2, if V1 is subgraph V2, prune V1. Guaranteed to find 
    all V2. Function allows to not prune those subgraphs which include node ids defined 
    by parameter select_ids.
    
    Returns:
    Vs_hat_pruned (list): list containing remaining candidate 
                          graphlets as lists of node ids
           pruned (list): list containing pruned candidate 
                          graphlets as lists of node ids
    Notes:
    ------
    Worst-case running time is O(n^2), but on average the running time is quasi-linear
    as the implementation explores the maximal graphlet candidates starting first from 
    the longest graphlets. 
    """
    n = U.shape[0]
    sel = np.array([True]*n)
    lens = {i:[] for i in range(k_min, k_max+1)}
    for i in range(n):
        Ui = U[i,:]
        u = frozenset(Ui[Ui >= 0])
        lens[len(u)].append((i,u))
        
    for li in range(k_min, k_max+1):
        for i,u_i in lens[li]:            
            for lj in range(li+1,k_max+1)[::-1]:
                for j,u_j in lens[lj]:
                    if u_i.issubset(u_j):
                        sel[i] = False
                        break
                if not sel[i]:
                    break
    return U[sel,:], U[sel == False,:]

def prune(A: np.array, U: np.array, k_min: int, k_max: int, delta_min: float,
          delta_max: float, force_select: list=[], n_sel: int=None,
          remove_small_seeds=True, verbose=False, weakly_connected=False):
    """Prune candidate graphlets using a multi-step pruning pipeline.
    
    Steps:
        1. keep strongly connected graphlets
        2. keep only graphlets that are maximal sets 
        3. select up to n_sel top ranking graphlets based on tau scores 
           with a guarantee that each node will be included in at least 
           p graphlets
        
    Returns:
    --------
    Vs_hat_pruned (np.array): array of shape (n, k_max) containing remaining 
                              candidate graphlets as rows.
    """ 
    t0p = process_time()
    n = U.shape[0]
    U_pruned = U
           
    if not weakly_connected:
        if verbose: 
            t0 = process_time()
            print(':: Pruning 1/3: Selecting strongly connected graphlets')
        
        U_pruned = strongly_connected(A, U_pruned)
        if verbose: 
            print(':: * Number of graphlets after ' \
                          'selection: {}'.format(U_pruned.shape[0]))    
            td = process_time() - t0
            print(':: (@ {})\n'.format(timedelta(seconds=td)))
                 
    # Prune subgrahps
    if verbose: 
        t0 = process_time()
        print(':: Pruning 2/3: Reducing subgraphs to ' \
                      'their largest supergraph-graphlets')
    
    U_pruned, pruned_s = prune_subgraphs(U_pruned, k_min, k_max)
    
    if verbose: 
        print(':: * Number of graphlets after subgraph ' \
                      'pruning: {}'.format(U_pruned.shape[0]))
    
        td = process_time() - t0
        print(':: (@ {})\n'.format(timedelta(seconds=td)))
    
    if n_sel is not None:
        if n_sel < U_pruned.shape[0]:
            if verbose:
                t0 = process_time()
                print(f':: Pruning 3/3: Selecting {n_sel} top ranking ' \
                      'graphlets')
                print(':: Computing tau scores...')
            
            taus, _ = graphlet_scores(U_pruned, A)
            if verbose: print(':: Sorting based on tau scores...')
            idxs = np.argsort(taus)[::-1]
            _, u_sel = binary_search_p(U_pruned[idxs,:], A.shape[0], n_max=n_sel, 
                                       tol=int(n_sel*0.1), verbose=True)
            U_pruned = U_pruned[u_sel,:]
            
            if verbose: 
                td = process_time() - t0
                print(':: (@ {})\n'.format(timedelta(seconds=td)))
                
                print(':: Pruning ready, {} graphlets selected'
                      .format(U_pruned.shape[0]))
                td = process_time() - t0p
                print(':: Total elapsed time @ {}\n'.format(timedelta(seconds=td)))
        else:
            if verbose: print(':: n_sel >= n, skipping ranked selection.')
                
    return U_pruned