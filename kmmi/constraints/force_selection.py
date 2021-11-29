import numpy as np
from numba import *
from numba.typed import Dict
from numba.types import bool_

import timeit
from time import process_time

@njit
def to_dependency_network(U: np.array, force_select: np.array):
    """Form a dependency network such that the set of nodes S contains all
    graphlets and 
    graphlets s,k \in U are connected if they share a force selected node 
    v in their intersection. Edge weight $w$ is defined as 
    $$ w_{sk} = |v \cap s \cap k|$$.
    """
    n = len(U)
    n_fs = len(force_select)
    
    E_const = np.zeros((n,n))
    fs_map = np.zeros((n, n_fs), dtype=bool_)
    fs_idsn = Dict()
                      
    for k, v in zip(force_select, np.arange(n_fs)):
        fs_idsn[k] = v
    
    for i, Si in enumerate(U):
        fs_Si = set(force_select) & set(Si)
        if len(fs_Si) > 0:
            for u in fs_Si:
                fs_map[i,fs_idsn[u]] = True
                      
    for i,Si in enumerate(U):
        for j,Sj in enumerate(U):
            if i != j:
                w = len(set(Si) & set(Sj))
                if w > 0:
                    E_const[j,i] = w
    
    idxs = np.array([(E_const[i,:] > 0).sum() != 0 
                     for i in range(E_const.shape[0])])
    
    E_const = E_const[idxs,:][:,idxs]
    fs_map = fs_map[idxs,:]
    return U[idxs,:], E_const, fs_map, idxs

@njit
def maximal_independent_set(A: np.array) -> np.array:
    """Sample maximal independent set (numpy implementation) using a 
    greedy strategy.
    
    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximal
    independent set is an independent set such that it is not possible
    to add a new node and still get an independent set.
    
    Parameters
    ----------
    A : weighted adjacency matrix for the dependency network
    
    Returns
    -------
    s : indices of nodes in the found maximal independent set
    """
    n = A.shape[0]
    D = np.array([False for _ in range(n)])
    I = np.array([False for _ in range(n)])
    seed_node = np.random.choice(n)
    neighbors = np.where(A[seed_node,:])[0]
    D[seed_node] = I[seed_node] = True
    D[neighbors] = True
    while np.sum(D) < n:
        node = np.random.choice(np.where(D != True)[0])
        I[node] = True
        node_neighbors = np.where(A[node,:])[0]
        D[node_neighbors] = True
        D[node] = True
    s = np.where(I)[0]
    return s 

def sample_fs_configurations(A: np.array, U, n_v, fs, fs_map, target_time=30, adaptive=True,
                             tol=0.99, n_min_target=10000, verbose=False) -> list:
    """Sample maximal independent sets to determine a configuration of force selected nodes
    that guarantees near-minimal overlap.
    
    Employs adaptive enumeration, where the run length is determined by target_time
    parameter (and n_min_target sets the absolute minimum requirement).
    
    Parameters
    ----------
    A : weighted adjacency matrix of the input network
    fs : set of nodes that each graphlet configuration is required to contain
    fs_map : mapping of network nodes to graphlets (see `to_dependency_network` method)
    target_time : max time threshold for the run (unit in seconds)  
    n_min_target : minimum number of configurations that should be generated
    
    Returns
    -------
    ss (list of lists): configurations
    fs (np.array): updated set of force selected nodes
    Notes
    -----
    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximal
    independent set is an independent set such that it is not possible
    to add a new node and still get an independent set.
    """
    map_fs = lambda x:  np.unique(np.where(fs_map[x,:])[1])
    
    t = timeit.timeit(lambda: maximal_independent_set(A), number=100)
    n_sample = np.max([np.int32(1 / (t / 100)), 1000])
    rng = np.random.default_rng()
    u_sel = np.arange(U.shape[0])
    n_fsa = len(fs)
    if verbose: 
        print(':: Sampling configurations of s-nodes for force selection') 
        print(':: --> Targeting {:.0f}s, threshold set @ {:.2f}%'
              .format(target_time, tol*100))
    
    enums = set()
    ls_exs = []
    
    c = np.zeros(len(fs), dtype=int)
    removed = []
    i = n_ss0 = exs0 = i0 = i0a = 0
    t0 = pt0 = process_time()
    while pt0 - t0 < target_time or len(enums) < n_min_target:
        for _ in range(n_sample):
            s = maximal_independent_set(A) 
            fs_idxs = map_fs(s)
            c[fs_idxs] += 1
            i+=1
            l_so = len(set(fs_idxs))
            if l_so == len(fs):
                s_prime = u_sel[s]
                enums.add(frozenset(s_prime))
            else:
                ls_exs.append(l_so)
                
            if verbose:
                pt0, n_ss0, i0a = __output_stats(fs, enums, i, i0a, c, t0, pt0, n_ss0)
        
        # EVALUATE
        exs_frac = (len(ls_exs)-exs0) / (i - i0)
        if exs_frac > tol:
            if not adaptive:
                raise Exception('Configurations that satisfy the complete set of ' \
                                'force selected nodes appear too infrequently. ' \
                                'Either remove the problematic nodes or enable '\
                                'adaptive running mode by setting `adaptive=True`.')
            
            fs, c, fs_map, A, u_sel, removed = __drop(fs, c, fs_map, A, u_sel, 
                                                      removed, verbose)
            i0 = i
            exs0 = len(ls_exs)
            enums = set()
            if verbose:
                print(':: Discarded {:.2f}% of the configurations'.format(100*exs_frac))
                
    tdelta = process_time() - t0
    if verbose:
        print(':: GENERATION STATS:\n* {} generated\n* {} accepted\n' \
              '* {} discarded\n* {:.2f}% (+/-) {:.2f} of the FS nodes included on avg ' \
              '\n* elapsed time: {:.2f}s.'.format(i,i-len(ls_exs),len(ls_exs),
                                                           np.mean(ls_exs) / n_fsa*100,
                                                           np.std(ls_exs) / n_fsa*100,
                                                           tdelta))
        print(':: Found {} unique configurations, {} ({:.0f}%) force selected nodes were ' \
              'removed: {}'.format(len(enums),len(removed),len(removed)/n_fsa*100,removed))

    ss = [[v for v in e] for e in enums]
    return ss, fs

def __drop(fs, c, fs_map, A, u_sel, removed, verbose):
    """Drop the worst performing node in the current force selected set by
    occurrence in the generated configurations.
    """
    c_argsort = np.argsort(c)
    drop = c_argsort[0]
    keep = c_argsort[1:]
    if verbose:
        print(':: Node', fs[drop],'removed.')
    idxs = np.unique(np.where(fs_map[:, keep])[0])
    A = A[idxs,:][:,idxs]
    fs_map = fs_map[idxs,:][:,keep]
    c = c[keep]
    removed.append(fs[drop])
    fs = fs[keep]        
    u_sel = u_sel[idxs]
    return fs, c, fs_map, A, u_sel, removed

def __output_stats(fs, enums, i, i0a, c, t0, pt0, n_ss0):
    """Print iteration statistics in the `sample_fs_configurations` method.
    """
    if i % 10 == 0:
        pt1 = process_time()
        if pt1 - pt0 > 10.0:
            n_ss = len(enums)
            rate_1_sec = np.round((i-i0a) / (pt1-pt0), 2)
            rate_2_sec = np.round((n_ss-n_ss0) / (pt1-pt0), 2)
            rate_3_sec = np.round(((i-i0a)-(n_ss-n_ss0)) / (pt1-pt0), 2)
            i0a = i
            n_ss0 = n_ss
            pt0 = pt1
            tdelta = int(pt1-t0)
            print(50*'-','\nITERATION:', i,' (', tdelta, 's )\nF-node | present %')
            for ii in range(len(fs)):
                print('  ',ii,'    ', np.round(c[ii] / i * 100, 2))
            print('* Rate (all):      ', rate_1_sec, '/ s (', rate_1_sec * 3600,'/ h)')
            print('* Rate (accepted): ', rate_2_sec, '/ s (', rate_2_sec * 3600,'/ h)')
            print('* Rate (discarded):', rate_3_sec, '/ s (', rate_3_sec * 3600,'/ h)')
    return pt0, n_ss0, i0a