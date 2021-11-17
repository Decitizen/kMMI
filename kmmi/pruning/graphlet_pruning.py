import numpy as np
import networkx as nx
from numba import njit
from itertools import chain

from kmmi.enumeration.graphlet_enumeration import *
from kmmi.utils.utils import to_numpy_array

overlap_coefficient = lambda A,B: len(A & B) / np.min([len(A),len(B)])

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

def prune_by_density(A: np.array, U: np.array, ds: np.array=None, 
                     delta_min: float=0.7, delta_max: float=1.00): 
    """Prune all subgraphs Gs s.t. delta_min <= density(Gs) <= delta_max"""
    if ds is None:
        taus, ds = graphlet_scores(U, A)
    d_sel = (delta_min <= ds) & (ds <= delta_max) 
    d_unsel = (delta_min > ds) | (ds > delta_max)    
    assert d_sel.sum() > 0,  "All graphlets were pruned; " \
                             "selected density range may be too narrow, " \
                             "lower the delta_min or increase the delta_max " \
                             "to relax the requirement."
    return U[d_sel,:], U[d_unsel, :]

def prune_subgraphs(U: np.array, k_min: int, k_max: int):
    """For two node sets V1 and V2, if V1 is subgraph V2, prune V1. Guaranteed to find all V2. 
    Allows not pruning subgraphs that include node ids defined by parameter select_ids.
    
    Returns:
    Vs_hat_pruned (list): list containing remaining candidate graphlets as lists of node ids
    pruned (list): list containing pruned candidate graphlets as lists of node ids
    
    Note: In practice running time is quasi-linear.
    """
    n = U.shape[0]
    found = False
    u_sel = set()
    u_unsel = set()
    # Init dict with length classes for the node sets
    lens = {i:[] for i in range(k_min, k_max+1)}
    for i in range(n):
        Ui = U[i,:]
        u = Ui[Ui >= 0]
        lens[len(u)].append((i,frozenset(u)))

    # Start from shortest graphlets
    for li in range(k_min, k_max+1):
        for i,u_i in lens[li]:
            # Compare against other sets of graphlet lengths start 
            # from the longest, break if found, decrease l
            for lj in range(li+1,k_max+1)[::-1]:
                for j,u_j in lens[lj]:
                    # Stop criteria
                    if u_i.issubset(u_j):
                        u_unsel.add(i)
                        u_sel.add(j)
                        # Set stop flag
                        found = True
                        break
            if not found:
                u_sel.add(i)
            else:
                found = False
                break
                
    idxs = list(u_sel)
    return U[idxs,:], U[list(u_unsel),:]

def select_by_scores(U: np.array, A: np.array, taus_map: dict=None, 
                     n_sel: int=20000) -> list:
    """Return n_sel graphlets with highest scoring function values 
    as a list of lists. 
    """
    taus = np.apply_along_axis(lambda x: taus_map[frozenset(x)], 1, U)
    idxs = np.argsort(taus)[::-1]
    return U[idxs,:][:n_sel,:], U[idxs,:][n_sel:,:]

def find_pruned_fs_graphlets(U: np.array, S: np.array, fs: list):
    """Select all graphlets s in S which include any of
    the force selected nodes K that are not in U."""
    F = set(fs)
    F_prime = F - set(U.ravel())
    idxs = []
    for i,s in enumerate(S):
        if len(F_prime & set(s)) > 0:
            idxs.append(i)
    return S[idxs,:], F_prime

def prune(G: nx.DiGraph, Vs_hat: list, k_min: int, k_max: int, delta_min: float,
          delta_max: float, force_select: list=[], n_sel: int=None,
          remove_small_seeds=True, verbose=False):
    """Prune candidate graphlets based on density and set inclusivity. Finally, 
    if n_sel is defined, only n_sel candidates with highest graphlet weight 
    will be kept.
    
    Returns:
    Vs_hat_pruned (list): list containing remaining candidate graphlets as lists
    of node ids pruned (list): list containing pruned candidate graphlets as 
    lists of node ids.
    
    Note: In practice running time is quasi-linear.
    """ 
    A = nx.to_numpy_array(G, nodelist=G.nodes)
    node_map = dict(zip(G.nodes(), np.arange(len(G.nodes))))
    Vs_hat = [[node_map[u] for u in v] for v in Vs_hat]
    fs = [node_map[u] for u in force_select]
    U = to_numpy_array(Vs_hat)
    pruned = []
        
    # Compute tau values for graphlets 
    taus, ds = graphlet_scores(U, A)
    K = np.apply_along_axis(lambda x: frozenset(x), 1, U)
    taus_map = {}
    for i in range(U.shape[0]):
        taus_map[K[i]] = taus[i]
    
    # Prune by density
    if delta_min != 0.0: 
        if verbose: print(':: Pruning 1/2: Pruning graphlets by density')
        U_pruned, pruned_s = prune_by_density(A, U, ds, delta_min, delta_max)
        pruned.append(pruned_s)
        if verbose: print(':: * Number of graphlets after ' \
                          'density pruning: {}\n'.format(U_pruned.shape[0]))
    # Prune subgrahps
    if verbose: print(':: Pruning 2/2: Reducing subgraphs to ' \
                      'their largest supergraph-graphlets')
    U_pruned, pruned_s = prune_subgraphs(U_pruned, k_min, k_max)
    pruned.append(pruned_s)
    if verbose: print(':: * Number of graphlets after subgraph ' \
                      'pruning: {}\n'.format(U_pruned.shape[0]))
    if n_sel is not None:
        U_pruned, pruned_s = select_by_scores(U_pruned, A, taus_map, n_sel)
        pruned.append(pruned_s)
        if verbose: print(':: * Number of graphlets after weight ' \
                          'selection: {}\n'.format(U_pruned.shape[0]))
    if len(fs) > 0:
        pruned = np.concatenate(pruned)
        pruned, not_included = find_pruned_fs_graphlets(U_pruned, pruned, fs)
        if len(pruned) > 0:
            n0 = U_pruned.shape[0]
            U_pruned = restore_pruned_graphlets(A, U_pruned, taus_map, pruned,
                                                not_included, fs, k_min, k_max, 
                                                remove_small_seeds=remove_small_seeds, 
                                                verbose=True)
            if verbose:
                print(':: Augmentation ready. Successfully restored ' \
                      '{} graphlets'.format(U_pruned.shape[0] - n0))
    return U_pruned

def graphlet_scores(U: np.array, A: np.array):
    """Compute graphlet scores for graphlets in U. 
    
    Graphlet scoring function is defined as \tau = ((s_rw + s_rho) / 2, 
    where s_rw is the normalized ranking order based on average edge 
    weight $$ \sum_{i,j in V_s} A_{ij} / n_e $$ such that A is the 
    weighted adjacency matrix of the subgraph G_s induced by the set 
    of nodes $s = \{u_1,u_2,...,u_n}$, and n_e is the number of 
    edges with non-zero weight in the induced subgraph G_s.
    """
    n = U.shape[0]
    ws = np.zeros(n)
    ds = np.zeros(n)
    for i in range(n):
        Ui = U[i,:]
        s = Ui[Ui >= 0]
        n_s = len(s)
        B = A[s,:][:,s]
        n_e = (B > 0.0).sum()
        ws[i] = B.sum() / n_e
        ds[i] = n_e / (n_s*(n_s-1))
    
    rank = (np.arange(n)+1) / n
    idxs = np.argsort(ws)
    taus = (rank[idxs] + ds[idxs]) / 2
    C = np.column_stack((idxs,taus))
    C = C[C[:,0].argsort()]
    return C[:,1], ds

def select_by_weight(graphlets: list, A: np.array, 
                     node_map: dict, n_sel: int=20000) -> list:
    """Return n_sel graphlets with highest weight as a list of lists. 
    Graphlet weight is defined as element-wise sum of the adjacency 
    matrix A: $$ \sum_{i,j in V_s} A_{ij} $$ where A is the weighted 
    adjacency matrix of the subgraph induced by the set of nodes 
    $u \in s$ and where $s$ is the graphlet.
    """
    s_ws = []
    for s in graphlets:
        s_prime = [node_map[u] for u in s]
        s_w = A[s_prime,:][:,s_prime].sum()
        s_ws.append(s_w)
        
    scores, graphlets = zip(*sorted(zip(s_ws,graphlets))[::-1])
    return list(graphlets[:n_sel]), list(graphlets[n_sel:])

def sort_and_restore(Cv, n_restore, overlap_threshold, verbose=False):
    sorted_taus = sorted(list(Cv), reverse=True)
    print(sorted_taus)
    restored = [sorted_taus[0][1]]
    restored_taus = [sorted_taus[0][0]]
    for tau, u in sorted_taus:
        if np.all([overlap_coefficient(u, v) < overlap_threshold 
                   for v in restored]):
            restored.append(u)
            restored_taus.append(tau)
            if len(restored) == n_restore:
                break
    return restored, restored_taus

def restore_pruned_graphlets(A: np.array, U: np.array, taus_map: dict, pruned: list,
                             not_included: set, select_ids: list, k_min: int, k_max: int, 
                             n_restore: int=5, overlap_threshold: float=0.35, 
                             verbose=False, remove_small_seeds=False):
    """For each node v for which all graphlets were pruned during the pruning phase, restore 
    number of pruned graphlets where number is defined by n_restore attribute. See compute_tau 
    method for definition of the selection criteria.
    
    Returns:
    U (np.array):  with restored graphlets, 
                   list of lists of node ids 
    """
    ## Add back pruned graphlets as necessary
    print(':: {} colored nodes were pruned.'.format(len(not_included)))
    restored_all = []
    # Collect all graphlets based on v's group membership, compute tau
    if len(pruned) > 0:
        
        C = {v:set() for v in not_included}
        for v in not_included:
            for i,u in enumerate(pruned):
                if v in u:              
                    u_set = frozenset(u)
                    C[v].add((taus_map[u_set], u_set))
                    
        for v, Cv in C.items():
            if len(Cv) > 0:
                restored, r_taus = sort_and_restore(Cv, n_restore, 
                                                    overlap_threshold, 
                                                    verbose)
                for u in restored:
                    restored_all.append(u)
                if verbose:
                    n_r = len(restored)
                    print(f':: {len(C[v])} graphlets with node {v}, {n_r}' \
                          ' nodes restored, tau values:', r_taus)
            else:
                if remove_small_seeds:
                    select_ids.remove(v)
                    if verbose:
                        print(':: No suitable graphlets found for node {},' \
                              ' removing node from selection'.format(v))
                else:
                    exp_msg = ':: No graphlet candidates of size range '\
                              '{}-{} for node {}.'.format(k_min, k_max, v)
                    raise SeedNodeError(exp_msg)
    else:
        exp_msg = ':: 0 pruned graphlet candidates found, cannot add ' \
                  'graphlets for all force-selected nodes.'
        raise SeedNodeError(exp_msg)
    
    U_aug = np.array([list(s) for s in restored_all])
    return np.concatenate((U, U_aug))

