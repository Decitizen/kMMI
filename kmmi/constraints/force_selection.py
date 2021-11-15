import numpy as np
from numba import *

@njit
def to_dependency_network(U: np.array, force_select: np.array):
    """Form a dependency network such that nodes represent graphlets and 
    graphlets s,k \in U are connected if they share a force selected node 
    v in their intersection. Edge weight $w$ is defined as 
    $$ w_{sk} = |v \cap s \cap k|$$.
    """
    n = len(U)
    n_fs = len(force_select)
    
    ol_fs_ids = set()
    E_const = np.zeros((n,n))
    fs_map = np.zeros((n, n_fs))
    fs_idsn = Dict()
    fs_idsn_rev = Dict()
                      
    for k, v in zip(force_select, np.arange(n_fs)):
        fs_idsn[k] = v
        fs_idsn_rev[v] = k

    for i, Si in enumerate(U):
        fs_Si = set(fs) & set(Si)
        if len(fs_Si) > 0:
            for u in fs_Si:
                fs_map[i,fs_idsn[u]] = 1
                      
    for i,Si in enumerate(U):
        for j,Sj in enumerate(U):
            if i < j:
                w = len(set(force_select) & set(Si) & set(Sj))
                if w > 0:
                    E_const[j,i] = w
                    ol_fs_ids.add(j)
                    ol_fs_ids.add(i)
                    
    return E_const, ol_fs_ids, fs_map, fs_idsn, fs_idsn_rev

def generate_fs_configuration(A, fs, fs_map, rng, verbose=False):
    """Generates configuration s by randomly picking new nodes until all force selected 
    nodes are in the set(s). Done by selecting a maximal independent set (MIS) in the 
    dependency network represented by weighted adjacency matrix A. If the size of the 
    size of the found MIS < size of the complete set of force selected nodes, then more
    nodes are picked by relaxing the independence assumption one edge weight unit at a time.
    """
    n = A.shape[0]
    I = np.array([False for _ in range(n)]) # Bool arr for keeping track of selected nodes
    D = np.array([True for _ in range(n)]) # Bool arr for keeping track of available nodes
    C = I.copy() # Bool arr for keeping track of nodes in the subgraph
    W = A.copy() # Weighted adjacency matrix
    
    while True:
        idx = rng.choice(len(fs))
        s = rng.choice(np.where(fs_map[:,idx])[0], size=1)
        if len(s) > 0: break
            
    s_set = set(np.where(fs_map[s,:])[0])
    I[s] = True
    
    while len(s_set) < len(fs):
        w_max = W.max()
        if w_max <= 0 or (W > 0).sum() == 0:
            raise Exception('No full configuration available.')
        
        fs_diff = np.array([u for u in set(fs) - s_set])
        fs_mapped = np.where(fs_map[:,fs_diff])[0]

        D[:] = True
        C[:] = False
        D[fs_mapped] = False
        C[fs_mapped] = True
        D[np.where(I)[0]] = C[np.where(I)[0]]= True
        W -= 1
        W[W < 0] = 0.0
        s_n = np.unique(np.where(W[s,:])[1])
        neighbors = np.where(C[s_n])[0]
        D[neighbors] = D[s] = True
        while (np.sum((D != True) & C) > 0) and (len(s_set) < len(fs)):
            node = rng.choice(np.where((D != True) & C)[0])
            node_set = set(np.where(fs_map[node,:])[0])
            if len(s_set | node_set) > len(s_set):
                node_neighbors = np.where(C[np.where(W[node,:])[0]])[0]
                I[node] = D[node] = D[node_neighbors] = True
                for u in node_set:
                    s_set.add(u)
            else:
                D[node] = True
                
    return np.where(I)[0]

def sample_fs_configurations(A: np.array, fs, fs_map, target_time=20, verbose=False):
    """Sample maximal independent sets to determine a configuration of force selected nodes
    that guarantees near-minimal overlap.
    
    Employs adaptive enumeration, where the run length is determined by target_time
    parameter (and N_MIN_TARGET set the absolute minimum requirement).
    
    Notes
    -----
    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximal
    independent set is an independent set such that it is not possible
    to add a new node and still get an independent set.
    
    """
    N_MIN_TARGET = 100
    rng = np.random.default_rng()
    fsa = np.arange(len(fs))
    if verbose: 
        print(':: Sampling configurations of s-nodes for force selection') 
        print(':: --> Targeting {:.0f}s...'.format(target_time))
    enums = set()
    t0 = process_time()
    i = exs = 0
    while process_time() - t0 < target_time or i < N_MIN_TARGET:
        try:
            s_aug = generate_fs_configuration(A, fsa, fs_map, rng, verbose=verbose)
            enums.add(frozenset(s_aug))
            i+=1
        except Exception as e:
            print(repr(e))
            exs += 1
            
    tdelta = process_time() - t0
    if verbose: 
        print(':: Ran {} iterations, {} exceptions, elapsed time: {:.2f}s.'.format(i,exs,tdelta))
        print(':: Found {} unique configurations.'.format(len(enums)))
    return [[v for v in e] for e in enums]