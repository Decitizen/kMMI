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
    for v in range(n):
        w = c = d = 0
        n_s = U[v,:].shape[0]
        for i in range(n_s):
            for j in range(n_s):
                if i == j: continue
                if U[v,i] != -1 and U[v,j] != -1:
                    Aij = A[U[v,i],U[v,j]]
                    w = w + Aij 
                    c += 1
                    if Aij > 0:
                        d += 1       
        ws[v] = w / c if c > 0 else 0
        ds[v] = d / c if c > 0 else 0
    return ws, ds

@njit
def select_nrank(U: np.array, A: np.array, Vs: np.array, p_min: int, p_max: int,
                 ptol: int=0.01,  n_iters: int=10, presorted=False, adaptive_p=True,
                 verbose=False) -> np.array:
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
    p_min : target number of graphlets to select per each seed node
    p_max : upper bound for the number of times each node can appear in the set of 
            selected graphlets 
    ptol : allowed tolerance parameter as a fraction of number of seed nodes
    presorted : set True if U rows are already in an ascending sorted order 
                from the most to the least optimal.
    adaptive_p : set True if p_max is allowed to be relaxed adaptively, this allows
                 finding the minimum p_max value such that all seed nodes are included 
                 at least p_min times

    Returns
    -------
    U_out : output graphlets, array of shape (n_sel, k_max) containing remaining 
        candidate graphlets as rows of node indices in the adjacency matrix of the 
        input network
    u_sel : boolean array of size U.shape[0] which determines which graphlets in 
            the original list of graphlets 
    C : counts of seed nodes in the set of selected output graphlets 
    p_max : updated number of times each node can appear in the set of selected 
            graphlets
    
    Notes
    -----
    Setting `verbose=True` will give useful information for diagnosing
    the run.
    """
    assert p_min <= p_max
    
    n = U.shape[0]
    k = U.shape[1]
    n_v = A.shape[0]
    n_vs = Vs.shape[0]
    
    if not presorted:
        if verbose: print(':: Computing tau scores...')
        taus, _ = graphlet_scores(U, A)
        if verbose: print(':: Sorting based on tau scores...')
        idxs = np.argsort(taus)[::-1]
        U = U[idxs,:]
        
    while True:
        for ii in range(n_iters):
            if ii == 0:
                U_idxs = np.arange(n)
            else:
                step = 100 if n > 1000 else 10
                U_idxs = __block_shuffle(n, step)
            if verbose: 
                print(':: * Targeting at least', p_min, \
                      ' graphlets per node\n\tPROGRESS:')
            count = 0
            stop_flag = False
            u_sel = np.array([False]*n)
            C = np.zeros(n_v)

            prcs = set([int(i) for i in (n * (np.arange(1,10) / 10))])
            
            for i in range(n):
                if verbose:
                    if i+1 in prcs:
                        prc = np.round((i+1) / n * 100, 1)
                        avg_fill = np.round(C[Vs].mean(), 2)
                        print(' \t*',prc,'% i:',i, \
                              '|', u_sel.sum(),'graphlets (', avg_fill, \
                              'per seed ) |', (C > 0).sum(), 'seeds')
                # Condition 1
                cond_one = False
                for j in range(k):
                    v = U[U_idxs[i],j]
                    if v != -1:
                        if C[v]+1 >= p_max:
                            cond_one = True
                            break
                if cond_one: continue
                u_sel[U_idxs[i]] = True
                # Count
                for j in range(k):
                    v = U[U_idxs[i],j]
                    if v >= 0:
                        C[v] += 1
                        # Condition 2
                        if C[v] == p_min:
                            count += 1
                            if count == n_vs:
                                stop_flag = True
                                break
                if stop_flag:
                    if verbose:
                        prc = np.round((i+1) / n * 100, 1)
                        print(':: Selection ready at iteration ',i,'(',prc,'%).')
                        print(':: Selected ', u_sel.sum(),'graphlets')
                    break
            if stop_flag: break
        
        if n_vs-count <= ptol*n_vs:
            if verbose:
                print('::', count, '/', n_vs, 'seed nodes were selected, ' \
                      'selection successful.')
                print(':: Final p range: [', p_min, ',', p_max, ']')
            break
        else:
            if verbose:
                print(':: Not all seed nodes could be selected,', \
                      np.sum(C>0), '/', n_vs, 'seed nodes were included in the selection, '\
                      'of which', count, 'nodes at least p_min times.')
                
            if adaptive_p and verbose:
                print(':: Relaxing p_max criteria to:', p_max+1)
                print(50*'--')
            if adaptive_p:
                p_max += 1
            else:
                break
                #raise Exception(':: Selection was unsuccessful.')

    return U[u_sel,:], u_sel, C, p_max

@njit
def __block_shuffle(n, step=10):
    """Generate approximate ordering of the U by blocking indeces of U into 
    blocks of size `step` and shuffling the order of the indeces in each block. 
    """
    U_idxs = np.arange(n)
    for i in np.arange(0,n,step):
        if i <= n - 2*step:
            Ui = U_idxs[i:i+step]
            np.random.shuffle(Ui)
            U_idxs[i:i+step] = Ui
        else:
            Ui = U_idxs[i:]
            np.random.shuffle(Ui)
            U_idxs[i:] = Ui
    return U_idxs

def binary_search_p(U: np.array, A: np.array, Vs: np.array, p_max: int, tol: float=0.1,
                    ptol: float=0.05,  n_max: int=5000, verbose=False) -> tuple:
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
        print(':: Initializing binary search for determining upper bound for p_min...')
        print(f':: Tolerance: {tol}')
    i = n_sel = 1
    while n_sel < n_max:
        p = 2**i
        p_max = np.max([p, p_max])
        _, idxs, _, p_max = select_nrank(U, A, Vs, p, p_max, verbose=verbose, 
                                         presorted=True, adaptive_p=False, ptol=ptol)
        n_sel = idxs.sum()
        i += 1

    if verbose: print(f':: Initial upper bound found for the p_min: {p}')
    d1 = np.inf
    S0 = idxs
    L, R = p/2, p
    while np.abs(d1) > tol*n_max:
        m = int((L + R) / 2)
        p_max = np.max([p, p_max])
        _, idxs, _, p_max = select_nrank(U, A, Vs, m, p_max, verbose=verbose, 
                                         presorted=True, adaptive_p=False, ptol=ptol)
        d1 = idxs.sum() - S0.sum()
        S0 = idxs    
        if idxs.sum() <= n_max:
            L = m
            if verbose: print(f':: * {m} ({idxs.sum()}) set as the lower bound')
        else:
            R = m
            if verbose: print(f':: * {m} ({idxs.sum()}) set as the upper bound')
    if verbose: print(':: Convergence succesful, final p_min: {} ({} selected)'
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

@njit
def overlap_selection(U: np.array, omega: float=0.5, n_max: int=5000, 
                      presorted=False, verbose=False):
    """Select at most n_max graphlets such that any pair (s_i, s_j) of graphlets
    will have at most omega overlap coefficient.
    
    Parameters
    ----------
    U : input graphlets as rows of an array with shape (n, k_max) where elements 
        are node indices in the adjacency matrix of the input network. Rows are 
        padded from the right with -1 for graphlet sizes < k_max
    omega : threshold parameter for maximum allowed overlap 
    n_max : maximum cap for how many graphlets can be selected
    presorted : set True if U rows are already in an ascending sorted order 
                from the most to the least optimal.
    
    Returns:
    --------
    S (list of sets): selected graphlets as list of sets
    
    Notes:
    ------
    Running time is positively correlated on both the allowed overlap (omega) 
    and n_max, decreasing both will result in reduced running times.
    
    Time complexity in worst-case is approximately O(mn_max^2), where m is 
    the constant cost of the set intersection operation for pair of graphlets. 
    However, note that as omega -> 0, the rate of growth of S will slow down, 
    and there is a regime of omega beyond which size of S will only be able to 
    reach a fraction of n_max.
    """
    if not presorted:
        if verbose: print(':: Computing tau scores...')
        taus, _ = graphlet_scores(U, A)
        if verbose: print(':: Sorting based on tau scores...')
        idxs = np.argsort(taus)[::-1]
        U = U[idxs,:]
        if verbose: print(':: Sorting completed...')        
        
    n = U.shape[0]
    S = []
    u_sel = np.array([False]*n)
    for i in range(n):
        ols = False
        si = set(U[i,:]) - set([-1])
        for sj in S:
            li, lj = len(si), len(sj)
            dnmr = li if li < lj else lj 
            if len(si & sj) / dnmr > omega:
                ols = True
                break
        if not ols:
            u_sel[i] = True
            S.append(si)
            if len(S) == n_max:
                break
    return S, u_sel

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