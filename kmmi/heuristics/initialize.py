import numpy as np
import networkx as nx
from numba import *
from kmmi.utils.utils import sub_sum
from kmmi.heuristics.neighborhood_search import ls_one_n_beam

@njit
def initialize_degree_vecs(A, H=None):
    n = A.shape[0]
    if H is not None:
        U = np.zeros(n, dtype=bool_)
        U[H] = True
        alpha = np.sum(A[:,U], axis=1, dtype=np.float64)
        beta = np.sum(A[:,U!=True], axis=1, dtype=np.float64)
    else:
        U = np.ones(n, dtype=bool_)
        alpha = np.sum(A, axis=1, dtype=np.float64)
        beta = np.zeros(n, dtype=np.float64)
    return U, alpha, beta

def __weighted_degree_rank(A, beta_ratio):
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
    return score_order, p_w

def init_solution_weighted_degree_ranking(A: np.array, k: int, beta_ratio: float=0.5):
    """Construct a k sized subgraph based on the degree rank order heuristic.
    """
    score_order, p_w = __weighted_degree_rank(A, beta_ratio)
    idxs = np.array(score_order[:k])
    H = np.zeros(A.shape[0], dtype=bool)
    H[idxs] = True
    return H, p_w

def init_solution_weighted_degree_ranking_fs(A: np.array, k: int, fss: list, 
                                             beta_ratio: float=0.25):
    """Construct a k sized subgraph based on the degree rank order heuristic
    taking into account the set of force selected nodes.
    """
    n = A.shape[0]
    H_fs = np.zeros(n, dtype=bool)
    idx = np.argmax([sub_sum(A, s) for i,s in enumerate(fss)])
    H_fs[fss[idx]] = True
    score_order, p_w = __weighted_degree_rank(A, beta_ratio)    
    
    H = np.zeros(n, dtype=bool)
    for v in score_order:
        if not H_fs[v]:
            H[v] = True
        if H.sum() + H_fs.sum() == k:
            break
    return H, H_fs, p_w

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
                    idxs = np.array(H)
                    H_w = sub_sum(A, idxs)
                    H = np.zeros(n, dtype=bool)
                    H[idxs] = True
                    return H, H_w
    
    raise Error("Didn't find k length initial configuration")
    
@njit
def init_solution_drop_initial(A, k):
    """Construct initial solution H using the drop initial initialization scheme;
    start with V = H and iteratively remove node that contributes least to the 
    sum A_ij for i,j in H, until k nodes are left in H.
    """
    n = A.shape[0]
    H = np.ones(n, dtype=bool_)
    alpha = np.sum(A, axis=1, dtype=np.float64)
    beta = np.zeros(n, dtype=np.float64)
    for ii in range(n-k):
        idx = np.argsort(alpha[H])[0]
        xj = np.where(H)[0][idx]
        H[xj] = False
        for i in range(n):
            if i != xj:
                alpha[i] = alpha[i] - A[i,xj]
                beta[i] = beta[i] + A[i,xj]
    return H, alpha, beta

def init_solution_drop_initial_fs(A, k, fss=None):
    """Construct initial solution H using the drop initial initialization scheme;
    start with V = H and iteratively remove node that contributes least to the 
    sum A_ij for i,j in H, until k nodes are left in H. Then we find the fs sequence
    H_fs such that the intersection of H_fs and H is maximum, i.e. argmax|H \cap H_fs|.
    If H_fs is contained completely within H, we return H, else we modify H such that
    rest of the graphlets s \in H_fs \setminus H will become included into H.
    """
    
    # 1. Get initial H using the standard drop initialization
    H, alpha, beta = init_solution_drop_initial(A, k)
    n = A.shape[0]
    n_fs = len(fss)
    
    max_ol = 0
    max_idx = -1
    max_fsi = None
    # 2. Find fs sequence with max overlap (largest intersection) with H
    for i in range(n_fs):
        fsi = np.zeros(n, dtype=bool)
        fsi[fss[i]] = True 
        fsi_ol = H & fsi
        n_ol = fsi_ol.sum()
        if n_ol > max_ol:
            max_ol = n_ol 
            max_idx = i
            max_fsi = fsi_ol
    
    # 3. If H_fs is corresponding to max_idx is not completely contained within H
    if len(set(fss[max_idx]) - set(np.where(H)[0])) != 0:
        H_fs = np.zeros(n, dtype=bool)
        
        # 3a. for overlapping cases modify H by removing the overlapping part
        if max_ol != 0:
            H_p = np.array(list(set(fss[max_idx])-
                           set(np.where(H)[0])))
            H[max_fsi] = False
            
        # 3b. for non-overlapping cases, go over the fs and select H_fs such that
        # its union is maximized (instead of intersection)
        else:
            max_w = 0
            for i in range(n_fs):
                fsi = np.zeros(n, dtype=bool)
                fsi[fss[i,:]] = True 
                fsi = H | fsi
                fsi_w = sub_sum(A, np.where(fsi)[0])
                if fsi_w > max_w:
                    max_w = fsi_w
                    max_idx = i
                    max_fsi = fsi
            H_p = fss[max_idx]

        H_fs[fss[max_idx]] = True

        # 4. Iterate through all nodes in H_fs \setminus H
        for xj in H_p:
            HC = H.copy()
            HC[H_fs] = False
            
            # Get node xi from H \setminus H_fs that least contributes to H 
            idx = np.argsort(alpha[HC])[0] 
            xi = np.where(HC)[0][idx]
            
            # Remove node xi from H, add xj instead
            H[xi] = False
            for y in range(n):
                if y != xj:
                    alpha[y] = alpha[y] - A[y,xi] + A[y,xj]
                    beta[y] = beta[y] + A[y,xi] - A[y,xj]
    else:
        # If fully contained
        H_fs = max_fsi
        H[H_fs] = False
    return H, H_fs, alpha, beta

def init_random_fs(A, k, fss):
    """Initialize H and H_fs using random initialization scheme s.t. H_fs configuration
    is picked first uniformly in random over list of configurations fss, after which
    k-|H_fs| nodes are randomly drawn from the set of V \ H_fs where V is the set of all
    nodes in the network.
    """
    n = A.shape[0]
    H = np.zeros(n, dtype=bool)
    H_fs = np.zeros(n, dtype=bool)
    idx = np.random.randint(len(fss))
    H_fs[fss[idx]] = True
    HC = np.where(H_fs == False)[0]
    idxs = np.random.choice(HC, size=k-H_fs.sum(), replace=False)
    H[idxs] = True
    return H, H_fs

def initialize_solution(A, A_beam, k, init_solution, init_mode, ls_tol, beta_ratio, 
                        find_maxima, one_in_k, verbose):
    """Initialize
    """
    n = A.shape[0]
    rsums = np.sum(A, axis=1)
    p_w = rsums / rsums.sum()
    if init_solution is None:
        if init_mode == 'drop-initial':
            H, _, _ = init_solution_drop_initial(A, k)
        elif init_mode == 'heaviest-edge':
            H, _ = init_solution_heaviest_edge_ranking(A, k)
        elif init_mode == 'weighted-deg':
            H, _ = init_solution_weighted_degree_ranking(A, k, beta_ratio=beta_ratio)
        else:
            H = np.zeros(n, dtype=bool)
            idxs = np.random.choice(n, k, replace=False)
            H[idxs] = True
            
        _, ao, bo = initialize_degree_vecs(A, H)
        H_w = sub_sum(A, np.where(H)[0])
        H, H_w, ao, bo = ls_one_n_beam(H, H_w, A, A_beam, alpha=ao, beta=bo, 
                                         tol=ls_tol, find_maxima=find_maxima, 
                                         one_in_k=one_in_k, verbose=verbose)    
    else:
        assert len(init_solution) == k
        H_w = sub_sum(A, H)
        H, ao, bo = initialize_degree_vecs(A, H)
        H_w = ao[H].sum() / 2
    
    return H, H_w, ao, bo, p_w
