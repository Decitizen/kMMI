import numpy as np
import networkx as nx
from numba import njit
from itertools import chain

from kmmi.enumeration.graphlet_enumeration import *

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

def density(s, A, node_map):
    """Computes density for grahplet s in a directed network
    represented by weighted adjacency matrix A."""
    assert len(s) >= 2, 'Graphlet is required to have at least 2 nodes.'
    
    n = len(s)
    si = [node_map[u] for u in s]
    rho = (A[si,:][:,si] > 0.0).sum() / (n*(n-1))
    return rho 

def prune_by_density(G: nx.DiGraph, Vs_hat: list, 
                     delta_min: float=0.7, 
                     delta_max: float=1.00, 
                     select_ids: list=[]):
    """Prune all subgraphs Gs s.t. delta_min 
       <= density(Gs) <= delta_max"""
    A = nx.to_numpy_array(G, nodelist=G.nodes) 
    node_map = dict(zip(G.nodes(), np.arange(len(G.nodes))))
    
    Vs_hat_pruned = []
    colored = []
    
    for s in Vs_hat:
        d = density(s, A, node_map)
        if delta_min <= d <= delta_max:
            Vs_hat_pruned.append(s)
        elif any([v in select_ids for v in s]):
            colored.append(s)
                    
    assert len(Vs_hat_pruned) > 0, "All graphlets were pruned; " \
                           "selected density range may be too narrow, " \
                           "lower the delta_min or increase the delta_max " \
                           "to relax the requirement."
    return Vs_hat_pruned, colored

def prune_subgraphs(Vs_hat: list, select_ids: list=[]):
    """For two node sets V1 and V2, if V1 is subgraph V2, prune V1. Guaranteed to find all V2. 
    Allows not pruning subgraphs that include node ids defined by parameter select_ids.
    
    Returns:
    Vs_hat_pruned (list): list containing remaining candidate graphlets as lists of node ids
    pruned (list): list containing pruned candidate graphlets as lists of node ids
    
    Note: In practice running time is quasi-linear.
    """
    pruned = []
    k_max = len(max(Vs_hat, key=len))
    found = False
    Vs_hat_pruned = set()
    
    # Init dict with length classes for the node sets
    lens = {i:[] for i in range(k_max+1)}
    for u in Vs_hat:
        lens[len(u)].append(frozenset(u))

    # Start from shortest graphlets
    for l in range(k_max+1):
        for u_leq in lens[l]:
            u_leq = frozenset(u_leq)
            
            # Compare against other sets of graphlet lengths start 
            # from the longest, break if found, decrease in size
            for i in range(l+1,k_max+1)[::-1]:
                if found: break
                for u_g in lens[i]:
                    # Stop criteria
                    if u_leq.issubset(u_g):
                        Vs_hat_pruned.add(u_g)
                        
                        # Keep track of pruned graphlets that include force-selected ids
                        if any([v in select_ids for v in u_leq]):
                            pruned.append(u_leq)
                        
                        # Set stop flag
                        found = True
                        break
                        
            if not found:
                Vs_hat_pruned.add(u_leq)
            else:
                found = False
    
    Vs_hat_pruned = [list(u) for u in Vs_hat_pruned]
    return Vs_hat_pruned, pruned

def prune(G: nx.DiGraph, Vs_hat: list, k_min: int, k_max: int, delta_min: float, 
          delta_max: float, weight: str='weight', force_select: list=[], 
          n_sel: int=None, remove_small_seeds=True, verbose=False):
    """Prune candidate graphlets based on density and set inclusivity. Finally, if n_sel 
    is defined, only n_sel candidates with highest graphlet weight will be kept.
    
    Returns:
    Vs_hat_pruned (list): list containing remaining candidate graphlets as lists of node ids
    pruned (list): list containing pruned candidate graphlets as lists of node ids
    
    Note: In practice running time is quasi-linear.
    """
    # Prune by density
    if delta_min != 0.0: 
        if verbose: print(':: Pruning 1/2: Pruning graphlets by density')
        Vs_hat_pruned, pruned_density = prune_by_density(G, Vs_hat, 
                                                         delta_min=delta_min, 
                                                         delta_max=delta_max, 
                                                         select_ids=force_select) 
        if verbose: print(':: * Number of graphlets after density pruning: {}\n'.format(len(Vs_hat_pruned)))
    
    # Prune subgrahps
    if verbose: print(':: Pruning 2/2: Reducing subgraphs to their largest supergraph-graphlets')
    
    Vs_hat_pruned, pruned_subgraphs = prune_subgraphs(Vs_hat_pruned)
    if verbose: print(':: * Number of graphlets after subgraph pruning: {}\n'.format(len(Vs_hat_pruned)))
    
    pruned_by_weight = []
    if n_sel is not None:
        A = nx.to_numpy_array(G, nodelist=G.nodes)
        node_map = dict(zip(G.nodes(), np.arange(len(G.nodes))))
        Vs_hat_pruned, pruned_by_weight = select_by_weight(Vs_hat_pruned, A, node_map, n_sel)
    
    pruned = pruned_subgraphs + pruned_density + pruned_by_weight
    not_included = set(force_select) - set(chain.from_iterable(Vs_hat_pruned))
    
    if len(not_included) > 0:
        n0 = len(Vs_hat_pruned)
        Vs_hat_pruned = restore_pruned_graphlets(G, Vs_hat_pruned, pruned, 
                                                 not_included, force_select,  
                                                 k_min, k_max, 
                                                 remove_small_seeds=remove_small_seeds, 
                                                 verbose=True)
        if verbose: 
            print(':: Augmentation ready. Successfully restored {} graphlets'.format(
                          len(Vs_hat_pruned) - n0))
    return Vs_hat_pruned

def compute_tau(s: list, A: np.array, node_map: dict):
    """Computes tau for graphlet s \in \hat G such that tau = s_rho + \langle w_s \rangle, 
    where s_rho is the density and \langle w_s \rangle is the average edge weight of s-induced 
    subgraph on G. 
    
    Returns:
    tau (float): 
    """    
    n = len(s)
    si = [node_map[u] for u in s]
    Asi = A[si,:][:,si]
    n_se = (Asi > 0.0).sum()
    w_s = Asi.sum() / n_se
    s_rho = n_se / (n*(n-1)) 
    tau = w_s + s_rho
    return tau

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

def compute_taus(G, not_included, pruned):
    A = nx.to_numpy_array(G, nodelist=G.nodes) 
    node_map = dict(zip(G.nodes(), np.arange(len(G.nodes))))
    
    C = {v:set() for v in not_included}
    taus = {}
    for v in not_included:
        for i,u in enumerate(pruned):
            if v in u:                        
                tau = taus[i] if i in taus else compute_tau(u, A, node_map)
                C[v].add((tau, frozenset(u)))
    return C, taus 

def sort_and_restore(Cv, n_restore, overlap_threshold, verbose=False):
    sorted_taus = sorted(list(Cv), reverse=True)
    restored = [sorted_taus[0][1]]
    restored_taus = []
    for tau, u in sorted_taus:
        if np.all([overlap_coefficient(u, v) < overlap_threshold 
                   for v in restored]):
            restored.append(u)
            restored_taus.append(tau)
            if len(restored) == n_restore:
                break
    return restored, restored_taus

def restore_pruned_graphlets(G: nx.DiGraph, Vs_hat: list, pruned: list, not_included: set, 
                             select_ids: list, k_min: int, k_max: int, n_restore: int=5, 
                             overlap_threshold: float=0.35, 
                             verbose=False, remove_small_seeds=False):
    """For each node v for which all graphlets were pruned during the pruning phase, restore 
    number of pruned graphlets where number is defined by n_restore attribute. See compute_tau 
    method for definition of the selection criteria.
    
    Returns:
    Vs_hat (list): original Vs_hat augmented with restored graphlets, 
                   list of lists of node ids 
    """  
    ## Add back pruned graphlets as necessary
    n_all = len(Vs_hat) 
    print(':: {} colored nodes were pruned.'.format(
        len(not_included))
    )
    restored = set()
    # Collect all graphlets based on v's group membership, compute tau
    if len(pruned) > 0:
        C, taus = compute_taus(G, not_included, pruned)

        for v, Cv in C.items():
            if len(Cv) > 0:
                restored, r_taus = sort_and_restore(Cv, n_restore, 
                                                           overlap_threshold, 
                                                           verbose)
                for u in restored:
                    Vs_hat.append(u)
                if verbose:
                    n_r = len(restored)
                    print(f':: {len(C[v])} graphlets with node {v}, {n_r} nodes restored, tau values:', r_taus)
            else:
                if remove_small_seeds:
                    select_ids.remove(v)
                    if verbose:
                        print(':: No suitable graphlets found for node {}, removing node from selection'.format(v))
                else:
                    exp_msg = ':: No graphlet candidates of size range {}-{} for node {}.'.format(k_min, k_max, v)
                    raise SeedNodeError(exp_msg)
    else:
        exp_msg = ':: 0 pruned graphlet candidates found, cannot add graphlets for all force-selected nodes.'
        raise SeedNodeError(exp_msg)
    return Vs_hat

