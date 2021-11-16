import numpy as np
import networkx as nx
from numba import *

def init_solution_weighted_degree_ranking(A: np.array, k: int, beta_ratio: float=0.25):
    """Construct a k sized subgraph based on the degree rank order heuristic.
    """
    n = A.shape[0]
    ws = np.sum(A, axis=0)
    ds = np.sum(A > 0, axis=0)
    _, s_ranking = zip(*sorted(zip(sorted(list(zip(np.arange(n), ws)), 
                                          key=lambda x: x[1]),range(n))))
    _, d_ranking = zip(*sorted(zip(sorted(list(zip(np.arange(n), ds)), 
                                          key=lambda x: x[1]),range(n))))
    
    s_ranking = (np.array(s_ranking) + 1) / n
    d_ranking = (np.array(d_ranking) + 1) / n
    beta_1 = beta_ratio
    beta_2 = 1.0 - beta_1
    scores = beta_1*s_ranking + beta_2*d_ranking
    p_w = scores / scores.sum()
    _, score_order = zip(*sorted(zip(scores, range(n)))[::-1])
    
    H = score_order[:k]
    H_w = A[H,:][:,H].sum() / 2
    
    return np.array(H), H_w, p_w

def init_solution_weighted_degree_ranking_fs(A: np.array, k: int, fss: list, beta_ratio: float=0.25):
    """Construct a k sized subgraph based on the degree rank order heuristic
    taking into account the set of force selected nodes.
    """
    H_fs = None
    if fss:
        sol_weight = lambda s: A[s,:][:,s].sum() / 2
        _, idx = max([(sol_weight(s),i) for i,s in enumerate(fss)])
        H_fs = fss[idx]

    n = A.shape[0]
    ws = np.sum(A, axis=0)
    ds = np.sum(A > 0, axis=0)
    _, s_ranking = zip(*sorted(zip(sorted(list(zip(np.arange(n), ws)), 
                                          key=lambda x: x[1]),range(n))))
    _, d_ranking = zip(*sorted(zip(sorted(list(zip(np.arange(n), ds)), 
                                          key=lambda x: x[1]),range(n))))
    
    s_ranking = (np.array(s_ranking) + 1) / n
    d_ranking = (np.array(d_ranking) + 1) / n
    beta_1 = beta_ratio
    beta_2 = 1.0 - beta_1
    scores = beta_1*s_ranking + beta_2*d_ranking
    p_w = scores / scores.sum()
    _, score_order = zip(*sorted(zip(scores, range(n)))[::-1])
    
    if fss:
        H = []
        for v in score_order: 
            if v not in H_fs:
                H.append(v)
            if len(H) + len(H_fs) == k:
                break
        H_w = sol_weight(H+H_fs)
    else:
        H = score_order[:k]
        H_w = sol_weight(H)
        
    return np.array(H), H_w, p_w, np.array(H_fs)

def init_solution_heaviest_edge_ranking(A, k, verbose=False):
    """Construct a k sized subgraph based on the heaviest k edge stubs heuristic.
    """
    # Sort indexes
    as_A = np.argsort(np.tril(A, -1), axis=None)[::-1]
    ind = np.unravel_index(as_A, A.shape)
    
    n = A.shape[0]
    H = []
    H_w = 0.0
    for i in range(n):
        x, y = ind[0][i], ind[1][i]
        for v in [x,y]:
            if v not in H:
                H.append(v)
                if len(H) == k:
                    if verbose: 
                        print(':: H has reached size of k: {}, breaking'.format(k))
                    H = np.array(H)
                    H_w = A[H,:][:,H].sum() / 2
                    return H, H_w
    
    raise Error("Didn't find k length initial configuration")