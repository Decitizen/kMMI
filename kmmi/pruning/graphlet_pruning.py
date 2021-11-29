import numpy as np
import networkx as nx
from numba import njit
from itertools import chain
from time import process_time
from datetime import timedelta

from kmmi.enumeration.graphlet_enumeration import *
from kmmi.utils.utils import to_numpy_array, mean_ndiag, overlap_coefficient
from kmmi.utils.autoload import *

def prune_by_density(U: np.array, A: np.array, ds: np.array=None, 
                     rho_min: float=0.7, rho_max: float=1.00) -> np.array: 
    """Prune all subgraphs G_s s.t. rho_min <= rho(G_s) <= rho_max"""
    if ds is None:
        _, ds = graphlet_scores(U, A)
    d_sel = (rho_min <= ds) & (ds <= rho_max) 
    assert d_sel.sum() > 0,  "All graphlets were pruned; " \
                             "selected density range may be too narrow, " \
                             "lower the rho_min or increase the rho_max " \
                             "to relax the requirement."
    return U[d_sel,:]

def strongly_connected(A: np.array, U: np.array) -> np.array:
    """Select all graphlets s in S which fulfill the definition of strongly 
    connected component in a directed network, namely that for each pair of 
    nodes u,v \in S there needs to exist at least one path for which u is 
    reachable from node v and vice versa.
    
    Parameters
    ----------
    U : input graphlets as rows of an array with shape (n, k_max) where elements 
        are node indices in the adjacency matrix of the input network. Rows are 
        padded from the right with -1 for graphlet sizes < k_max
    A : weighted adjacency matrix of the input network
    
    Returns:
    --------
    U : output graphlets, array of shape (n_sel, k_max) containing remaining 
        candidate graphlets as rows of node indices in the adjacency matrix
        of the input network.
    """
    
    SCC = lambda s: nx.is_strongly_connected(nx.DiGraph(A[s,:][:,s]))
    idxs = [SCC(s[s!=-1]) for s in U]
    return U[idxs, :]

@njit
def graphlet_scores(U: np.array, A: np.array) -> np.array:
    """Compute graphlet scores for graphlets in U. Graphlet scoring
    function is defined as such that, i!=j and
    $$\tau = \frac{1}{(n*(n-1))} \sum_{i,j \in s} A_{ij}$$
    where A is the weighted adjacency matrix of the subgraph G_s 
    induced by the set of nodes $s = \{u_1,u_2,...,u_n}$, and n is 
    the number of edges  in the induced subgraph G_s.
    
    Returns:
    --------
    ws (np.array): array of shape (U.shape[0]) containing the graphlets' weighted 
                   densities as rows such that indices are ordered corresponding
                   to the order of graplets in U.
    ds (np.array): array of shape (U.shape[0]) containing the graphlets' densities 
                   as rows such that indices are in order with corresponding
                   graplets in U.
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
def select_nrank(U: np.array, A: np.array, Vs: np.array, p: int, ptol: int=0.01,
                 presorted=False, strict_mode=True, verbose=False) -> np.array:
    """Selects p highest ranking graphlets per each node in the seed node set 
    Vs. This method assumes that U is already ordered in ascending ranking 
    order based on desired ranking criteria.
    
    Parameters
    ----------
    U : input graphlets as rows of an array with shape (n, k_max) where elements 
        are node indices in the adjacency matrix of the input network. Rows are 
        padded from the right with -1 for graphlet sizes < k_max
    A : weighted adjacency matrix of the input network
    Vs : set of seed nodes that will be selected for
    p : target number of graphlets to select per each seed node
    ptol : allowed tolerance (fraction of p) when using the non-strict mode
    presorted : set True if U rows are already in an ascending sorted order 
                from the most to the least optimal.
    strict_mode : controls how strict the selection behavior is, set False 
                  to relax the strict p guarantee to "approximately p", see
                  tol parameter for controlling the tolerance.

    Returns
    -------
    U : output graphlets, array of shape (n_sel, k_max) containing remaining 
        candidate graphlets as rows of node indices in the adjacency matrix
        of the input network.
    
    Notes
    -----
    Setting `verbose=True` will give useful information for diagnosing
    the run.
    """
    if not presorted:
        if verbose: print(':: Computing tau scores...')
        taus, _ = graphlet_scores(U, A)
        if verbose: print(':: Sorting based on tau scores...')
        idxs = np.argsort(taus)[::-1]
        U = U[idxs,:]
    
    if verbose: 
        print(':: * Targeting', p, 'graphlets per node\n\tPROGRESS:')
    
    Vss = set(Vs)
    n_vs = Vs.shape[0]
    count = 0
    flag = False
    n = U.shape[0]
    k = U.shape[1]
    u_sel = np.array([False]*n)
    C = np.zeros(A.shape[0])
    prcs = set([int(i) for i in (n * (np.arange(1,10) / 10))])
    for i in range(n):
        if verbose:
            if i+1 in prcs:
                prc = np.round((i+1) / n * 100, 1)
                avg_fill = np.round(C[Vs].mean(), 2)
                print(' \t*',prc,'% i:',i, \
                      '|', u_sel.sum(),'graphlets (', avg_fill, \
                      'per seed ) |', (C > 0).sum(), 'seeds')
        for j in range(k):
            Uidx = U[i,j]
            if Uidx in Vss:
                if C[Uidx] != p:
                    C[Uidx] += 1
                    count += 1
                    u_sel[i] = True
                    
                    if count == n_vs*p:
                        flag = True
                        break
        if flag:
            if verbose:
                prc = np.round((i+1) / n * 100, 1)
                print(':: Selection ready at iteration ',i,'(',prc,'%).')
                print(':: Selected ', u_sel.sum(),'graphlets')
            break
    
    if strict_mode:
        assert_msg = 'Selection not balanced, decrease p'
        assert count == n_vs*p, assert_msg
    else:
        assert p - np.mean(C[C > 0]) < ptol*p, assert_msg
    return U[u_sel,:], u_sel, C

def binary_search_p(U: np.array, A: np.array, Vs: np.array, tol: float=0.1, 
                    ptol: float=0.01,  n_max: int=5000, verbose=False) -> tuple:
    """Compute the best value of p parameter for the select_nrank function. Useful when 
    the number of graphlet candidates is larger than what the resources available for 
    running the downstream tasks that use the coarse-grained network. This implementation
    uses binary search to search for the limit p such that approximately n_max graphlets
    will be selected (tolerance is defined by tol parameter).
    
    Parameters
    ----------
    U : input graphlets as rows of an array with shape (n, k_max) where elements 
        are node indices in the adjacency matrix of the input network. Rows are 
        padded from the right with -1 for graphlet sizes < k_max
    A : weighted adjacency matrix of the input network
    Vs : set of seed nodes that will be selected for
    tol : tolerance of error for the n_max (fraction of n_max)
    ptol : tolerance of error for the selection method (fraction of p)
    
    Returns
    -------
    L (int) : optimal p value
    U (np.array) : output graphlets, array of shape (n_sel, k_max) containing remaining 
                   candidate graphlets as rows of node indices in the adjacency matrix
                   of the input network.
    """
    n = U.shape[0]
    n_v = A.shape[0]
    assert n > n_max, f'n: {n} > n_max: {n_max}, binary search isn\'t required'
    if n_max > 2e4: print(f'WARNING: n_max: {n_max} is higher than what the ' \
                          'pipeline has been benchmarked for.')

    if verbose:
        print(':: Initializing binary search for determining upper bound for p...')
        print(f':: Tolerance: {tol}')
    i = n_sel = 1
    while n_sel < n_max:
        p = 2**i
        _, idxs, _ = select_nrank(U, A, Vs, p, verbose=verbose, presorted=True, 
                                  strict_mode=False, ptol=ptol)
        n_sel = idxs.sum()
        i += 1

    if verbose: print(f':: Initial upper bound found for the p: {p}')
    d1 = np.inf
    S0 = idxs
    L, R = p/2, p
    while np.abs(d1) > tol*n_max:
        m = int((L + R) / 2)
        _, idxs, _ = select_nrank(U, A, Vs, m, verbose=verbose, presorted=True, 
                                  strict_mode=False, ptol=ptol)
        d1 = idxs.sum() - S0.sum()
        S0 = idxs    
        if idxs.sum() <= n_max:
            L = m
            if verbose: print(f':: * {m} ({idxs.sum()}) set as the lower bound')
        else:
            R = m
            if verbose: print(f':: * {m} ({idxs.sum()}) set as the upper bound')
    if verbose: print(':: Convergence succesful, final p: {} ({} selected)'
                      .format(m, S0.sum())) 
    return m, U[idxs,:]

def prune_subgraphs(U: np.array, k_min: int, k_max: int) -> np.array:
    """For two node sets V1 and V2, if V1 is subgraph V2, prune V1. Guaranteed to find 
    all V2.
    
    Returns:
    --------
    U (np.array): array of shape (n_sel, k_max) containing remaining candidate 
                  graphlets as rows of node indices in the adjacency matrix of
                  the underlying network.
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
    return U[sel,:]

def prune(A: np.array, U: np.array, Vs: np.array, k_min: int, k_max: int, 
          n_sel: int=None, verbose=False, weakly_connected=False) -> np.array:
    """Prune candidate graphlets using a multi-step pruning pipeline.
    
    Steps:
        1. keep strongly connected graphlets
        2. keep only graphlets that are maximal sets 
        3. select up to n_sel top ranking graphlets based on tau scores 
           with a guarantee that each node will be included in at least 
           p graphlets
        
    Returns:
    --------
    U (np.array): array of shape (n_sel, k_max) containing remaining candidate 
                  graphlets as rows of node indices in the adjacency matrix of
                  the underlying network.
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
    
    U_pruned = prune_subgraphs(U_pruned, k_min, k_max)
    
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
            _, U_pruned = binary_search_p(U_pruned[idxs,:], A, Vs, n_max=n_sel, 
                                       tol=int(n_sel*0.1), verbose=verbose)
            
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