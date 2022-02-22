from time import process_time
from datetime import timedelta as td
import numpy as np
from numba import *
from kmmi.utils.utils import __sub_sum
from kmmi.heuristics.initialize import *

@njit
def svns_score(H_w, Ho_w, H, Ho, k):
    return (H_w / Ho_w) + (k - (H & Ho).sum()) / k

@njit
def __update_degree_vecs(A, alpha, beta, xi, xj, inplace=False):
    alpha_p = alpha if not inplace else alpha.copy()
    beta_p = beta if not inplace else beta.copy()
    for y in range(A.shape[0]):
        if y != xj:
            alpha_p[y] = alpha[y] - A[y,xi] + A[y,xj]
            beta_p[y] = beta[y] + A[y,xi] - A[y,xj]
    return alpha_p, beta_p

@njit
def create_beam_array(A, A_as, w_thres):
    """Compute a beam array out of adjacency matrix A. In a beam array each row 
    i will contain the indexes of all connected nodes for node i in sorted order  
    based on the link weight."""
    
    n = A.shape[0]
    
    A_beam = np.zeros((n,n), dtype=np.int32) - 1 
    maxlen = 0
    for i in range(n):
        j = 0
        for k in A_as[i,:]:
            if A[i,k] >= w_thres:
                A_beam[i,j] = k
                j+=1
            else:
                if j > maxlen:
                    maxlen = j
                break
    
    return A_beam[:,:maxlen]

@njit
def create_beam_array_constant_width(A, A_as, w_thres):
    """Compute a beam array out of adjacency matrix A. In a beam array each row 
    i will contain the indexes of all connected nodes for node i in sorted order  
    based on the link weight."""
    
    #print('Beam width set')
    n_beam = 6
    n = A.shape[0]
    
    A_beam = np.zeros((n,n_beam)) - 1
    maxlen = 0
    for i in range(n):
        for j in range(n_beam):
            k = A_as[i,j]
            if A[i,k] > 0.0:
                A_beam[i,j] = k
                j+=1
            else:
                if j > maxlen:
                    maxlen = j
                break
                
    if maxlen < n_beam:
        A_beam = A_beam[:,:maxlen]
    
    return A_beam

@njit
def ls_one_n_beam(Uo, Uo_w, A, A_beam, alpha, beta, tol=0.0, 
                  find_maxima=False, one_in_k=False, verbose=False):
    """Computes local search in the 1-neighborhood of the Ho set such that 
    node is selected using beam criterion; the space of neighbors with heaviest 
    links in the 1-neighborhood of the current H nodes is searched first. First 
    improvement of the objective function is returned.
    """
    k = Uo.sum()
    n = A_beam.shape[1]
    
    Up_w = Uo_w
    f_prime = 0.0
    xip = xjp = -1
    
    u_idxs = np.where(Uo)[0]
    if not one_in_k:
        replace_ids = u_idxs.copy()
    if not find_maxima:
        np.random.shuffle(u_idxs)
        
    L = 0
    
    stop = False
    for i in range(k):
        if stop: break
        v = u_idxs[i]
        for j in range(n):
            if stop: break
            xj = A_beam[v,j]
            if xj != -1 and not Uo[xj]:
                if one_in_k:
                    replace_ids = np.random.choice(u_idxs, 1)
                for xi in replace_ids:
                    L += 1
                    delta_f = alpha[xj] - alpha[xi] - A[xi,xj]
                    if delta_f > f_prime:
                        Up_w = Uo_w + delta_f
                        f_prime = delta_f
                        xip = xi
                        xjp = xj
                        if verbose:
                            print(':: Improvement found: +', (delta_f))
                            print(':: Objective function value: ', Up_w,', iters: ', L)
                        if not find_maxima:
                            stop = True
                            break
    if Up_w == Uo_w:
        if verbose: print(':: No improvement found during local search.')
        return Uo, Uo_w, alpha, beta
    
    assert xip >= 0 and xjp >= 0
    alpha_p, beta_p = __update_degree_vecs(A, alpha, beta, xip, xjp)
    Up = Uo.copy()
    Up[xjp] = True
    Up[xip] = False
    return Up, Up_w, alpha_p, beta_p

@njit
def ls_one_n_beam_fs(Uo, Uo_fs, Uo_w, A, A_beam, alpha, beta, tol=0.0, 
                           find_maxima=False, one_in_k=False, verbose=False):
    """Computes local search in the 1-neighborhood of the Ho set such that 
    node is selected using beam criterion; the space of neighbors with heaviest 
    links in the 1-neighborhood of the current H nodes is searched first. First 
    improvement of the objective function is returned.
    """
    k1 = Uo.sum()
    k2 = Uo_fs.sum()
    n = A_beam.shape[1]
    
    # Keep track of best improvement
    Up_w = Uo_w
    f_prime = 0.0
    xip = xjp = -1
    
    u_idxs = np.where(Uo)[0]
    u_idxs_fs = np.where(Uo_fs)[0]
    
    if not one_in_k:
        replace_ids = u_idxs.copy()
    if not find_maxima:
        np.random.shuffle(u_idxs)
        
    L = 0
    
    stop = False
    for i in range(k1+k2):
        if stop: break
        v = u_idxs[i] if i < k1 else u_idxs_fs[i-k1] 
        for j in range(n):
            if stop: break
            xj = A_beam[v,j]
            if xj != -1 and not Uo[xj] and not Uo_fs[xj]:
                if one_in_k:
                    replace_ids = np.random.choice(u_idxs, 1)
                for xi in replace_ids:
                    L += 1
                    delta_f = alpha[xj] - alpha[xi] - A[xi,xj]
                    if delta_f > f_prime:
                        Up_w = Uo_w + delta_f
                        f_prime = delta_f
                        xip = xi
                        xjp = xj
                        if verbose:
                            print(':: Improvement found: +', (delta_f))
                            print(':: Objective function value: ', Up_w,', iters: ', L)
                        if not find_maxima:
                            stop = True
                            break
    if Up_w == Uo_w:
        if verbose: print(':: No improvement found during local search.')
        return Uo, Uo_w, alpha, beta
    
    assert xip >= 0 and xjp >= 0
    alpha_p, beta_p = __update_degree_vecs(A, alpha, beta, xip, xjp)
    Up = Uo.copy()
    Up[xjp] = True
    Up[xip] = False
    return Up, Up_w, alpha_p, beta_p

@njit
def ls_one_n(Ho, Ho_w, A, tol=0.0, verbose=False):
    """Computes local search in the 1-neighborhood of the Ho set in
    random ordering.
    """
    n = A.shape[0]
    k = len(Ho)
    Hc = np.array([v for v in range(n) if v not in Ho])
    np.random.shuffle(Hc)
    np.random.shuffle(Ho)
    
    L = 1
    for u in Hc:
        for v in Ho:
            H = np.array([vi if vi != v else u for vi in Ho])
            H_w = __sub_sum(A, H)            
            L += 1
            if verbose:
                if L % 10000 == 0:
                    print(':: Iterations: ', L)
  
            if H_w > Ho_w + tol*Ho_w:
                if verbose:
                    print(':: Improvement found: +', (H_w-Ho_w))
                    print(':: Objective function value: ', Ho_w,', iters: ', L)
                return H, H_w
            
    if verbose: print(':: No improvement found during local search.')
    return Ho, Ho_w

# Remove
@njit
def local_search_ovns_fs(Ho, Ho_fs, Ho_w, A, A_beam, verbose=False, 
                      tol=0.0, ls_tol=0.0, find_maxima=False):
    """Computes local search in the 1-neighborhood of the Ho set. Select between
    'first improvement' and 'best improvement' strategies with find_maxima parameter.
    (best improvement strategy finds the local maxima).
    """
    H_t0_w = 0.0
    n = A.shape[0]
    
    while Ho_w - H_t0_w > tol * Ho_w:
        H_t0_w = Ho_w
        Ho, Ho_w = ls_one_n_beam_fs(Ho, Ho_fs, Ho_w, A, A_beam, ls_tol, verbose)
        if not find_maxima:
            return Ho, Ho_w

    if verbose: print(':: Local maxima:', Ho_w, '\n')
    return Ho, Ho_w

def shake(A: np.array, Ho: np.array, k: int, p: int, alpha: np.array, beta: np.array,
          p_w: np.array=None, use_pref_attachment=False):
    """Implements the perturbation routine for the VNS (variable neighborhood search)
    H by randomly drawing p node ids from the H without replacement and replacing them 
    with p randomly drawn node ids from the complement of H.
    
    Parameter use_pref_attachment controls for the weighting of the random distribution. 
    If True allows for preferential weighting based on vector of probabilities. Defaults 
    to False which equals uniform distribution over all nodes in the network.
    """
    n = A.shape[0]
    HC = np.where(Ho == False)[0]
    xis = np.random.choice(np.where(Ho)[0], size=p, replace=False)
    xjs = np.random.choice(HC, size=p, replace=False, p=p_w[HC] / p_w[HC].sum() 
                           if use_pref_attachment else None)
    H = Ho.copy()
    H[xis] = False
    H[xjs] = True
    alpha_p = alpha.copy() 
    beta_p = beta.copy()
    for i in range(p):
        xi = xis[i]
        xj = xjs[i]
        alpha_p, beta_p = __update_degree_vecs(A, alpha_p, beta_p, 
                                               xi, xj, inplace=True)
    H_w = alpha_p[H].sum() / 2
    return H, H_w, alpha_p, beta_p

def shake_fs(A: np.array, Ho: np.array, Ho_fs: np.array, fss: list, n_ss: dict, k: int, 
             p: int, alpha: np.array, beta: np.array, verbose=False):
    """Implements the perturbation routine for the VNS when forced selection is used.
    
        Method has two different behaviors that are dependent on the p value and H_fs size:
        1. when p is smaller than the smallest available H_fs, only nodes in H are swapped.
        2. when p is at least the size of the smallest available H_{fs} configuration, 
           H_fs configuration and p - |H_fs| nodes from H are swapped.
    """
    assert k > Ho_fs.sum(), 'size of H_fs is larger than k, try increasing k'
    n = A.shape[0]
    c_min = min(n_ss.keys())
    c_max = max(n_ss.keys())
    
    if p < c_min and p > Ho.sum():
        p = np.random.randint(c_min, c_max)
    if p >= min(n_ss.keys()):
        H, H_fs, xis, xjs = __replace_in_hhfs(Ho, Ho_fs, fss, n_ss, k, n, p, verbose)
    else:
        H, H_fs, xis, xjs = __replace_in_h(Ho, Ho_fs, p)
    
    alpha_p = alpha.copy()
    beta_p = beta.copy()
    for i in range(len(xis)):
        xi = xis[i]
        xj = xjs[i]
        alpha_p, beta_p = __update_degree_vecs(A, alpha_p, beta_p, 
                                               xi, xj, inplace=True)
    Hfs_len = (H | H_fs).sum()
    assert Hfs_len == k, f'H combined size {Hfs_len} does not match with k={k}'
    assert (H & H_fs).sum() == 0, 'There is overlap in the selection\n * : ' \
                                          '{}'.format(set(H_fs) & set(H))
    return H, H_fs, alpha_p, beta_p

@njit
def __replace_in_h(Ho, Ho_fs, p):
    """Replace p nodes in Ho by uniformly drawn sample from the 
    complement of Ho & Ho_fs."""
    H = Ho.copy()
    H_fs = Ho_fs.copy()
    HC = np.where((Ho | Ho_fs) == False)[0]
    xis = np.random.choice(np.where(Ho)[0], size=p, replace=False)
    xjs = np.random.choice(HC, size=p, replace=False)
    H[xis] = False
    H[xjs] = True  
    return H, H_fs, xis, xjs

def __replace_in_hhfs(Ho, Ho_fs, fss, n_ss, k, n, p, verbose=False):
    """Replace p nodes in both Ho and Ho_fs with uniformly drawn sample from the 
    complement of Ho & Ho_fs."""
    H = np.zeros(n, dtype=bool)
    while True:
        H_fs = np.zeros(n, dtype=bool)
        idxs = fss[__sample_up_to_class_p(p, n_ss)]
        H_fs[idxs] = True
        pp = len(idxs) - (H_fs & (Ho | Ho_fs)).sum()
        if p - pp + len(idxs) <= k:
            break
            
    if verbose: print(f':: Picked new fs node configuration of length {len(idxs)}')
    if verbose: print(f':: Number of unique new nodes {pp}')
    p_diff = p - pp
    if p_diff > 0:
        if verbose: print(f':: p_diff > 0, adding {p_diff} new unique nodes to H')
        HC = np.where((Ho | H_fs | Ho_fs) == False)[0]
        idxs = np.random.choice(HC, size=p_diff, replace=False)
        H[idxs] = True
        pp = pp + p_diff
        assert (H & H_fs).sum() == 0
        assert k == pp + (k-p)
    
    k_diff = k - (H | H_fs).sum()
    if k_diff > 0:
        if verbose: print(f':: k_diff > 0, adding {k_diff} nodes from Ho to H')
        Hu = set(np.where(H | H_fs)[0])
        HC = set(np.where(Ho)[0]) - Hu
        if len(HC) < k_diff:
            HC |= (set(np.where(Ho_fs)[0]) - Hu)
            if len(HC) < k_diff:
                HC = np.where((Ho | Ho_fs | H | H_fs) == False)[0]
        idxs = np.random.choice(list(HC), size=k_diff, replace=False)
        assert len(set(idxs) & set(np.where(H | H_fs)[0])) == 0
        H[idxs] = True
    
    xis = list(set(np.where(Ho | Ho_fs)[0]) - set(np.where(H | H_fs)[0]))
    xjs = list(set(np.where(H | H_fs)[0]) - set(np.where(Ho | Ho_fs)[0]))
    assert (H | H_fs).sum() == k, f'combined size is {(H | H_fs).sum()} vs {k}'
    assert (H & H_fs).sum() == 0
    assert len(xis) == len(xjs), 'Numbers of removed nodes ({}) and ' \
                                 'added ({}) nodes do not match.' \
                                 .format(len(xis),len(xjs))
    return H, H_fs, xis, xjs
        
def __sample_up_to_class_p(p, n_ss):
    assert p >= min(n_ss.keys()), 'p should be >= shortest length class in n_ss'
    # Pick randomly from classes of length <= p
    keys_sorted = np.array(sorted(n_ss.keys())) 
    keys_idxs = keys_sorted[keys_sorted <= p]
    lens = [len(n_ss[idx]) for idx in keys_idxs]
    idx = np.random.choice(np.sum(lens), replace=False)
    idx_t = [(i,idx - sum(lens[0:i])) for i in range(len(lens)) 
             if idx < np.sum(lens[0:i+1])][0]
    out_idx = n_ss[keys_idxs[idx_t[0]]][idx_t[1]]
    return out_idx

def __to_len_classes(ss):
    n_ss = {}
    for i,s in enumerate(ss):
        n_s = len(s)
        if n_s not in n_ss:
            n_ss[n_s] = []
        n_ss[n_s].append(i)
    return n_ss
        
def OVNS(k: int, A: np.array, k_lims: tuple, k_step: int=1, timetol: int=300, ls_tol: float=0.0, 
         ls_mode='best', use_pref_attachment=False, init_mode='drop-initial', beta_ratio: float=0.5, 
         seed: int=None, max_iter: int=100000, max_iter_upd: int=1000000, w_quantile=0.01, 
         init_solution: np.array=None, theta: float=0.06, svns = False, one_in_k=False, verbose=False): 
    """Variable neighborhood search heuristic for the HkSP. 
    
    Parameters
    ----------
    k : value of k in HkS (number of treatment groups)
    A : weighted adjacency matrix of the input network
    k_lims : range for search depth
    k_step : defines how many search depth steps to increment at once after failing update
    timetol : set the target time in seconds for the run
    ls_tol : set a tolerance for update (update is approved if improvement over current 
             best solution is at least ls_tol)
    ls_mode : define search mode, options are `best` and `first`, first being the default
    use_pref_attachment : define if perturbation will be weighted based on degree or uniformly
    init_mode : define initialization strategy ('heaviest-edge','weighted-deg','random')
    beta_ratio : sets the weights for the `weighted-deg` initialization strategy in range
                 [0,1.0] where 0 gives all weight to degree ranking and 1 to weighted 
                 degree ranking.
    seed : seed for the random number generator (note: due to lack of numba support for numpy, 
           some numpy.random library calls don't allow seed input currently)
    w_quantile : define the quantile of edge weights that will be explored during each local 
                 neighorhood update iteration, lower values in increase the
    max_iter : maximum number of iterations after last successful update
    
    Returns
    -------
    Ho : selected solution graphlets as row indeces in the adjacency matrix
    Ho_w : value of the objective function for the solution
    """
    n = A.shape[0]
    assert ls_mode in ['first','best'], "Invalid local search mode; choose either 'first' or 'best'" 
    assert init_mode in ['heaviest-edge','weighted-deg','random','drop-initial']
    assert 0.0 < w_quantile <= 1.0, 'Invalid value, select value from range (0.0,1.0]'
    assert k <= n, 'Input k value is greater than n; select k such that k <= n'
    assert A[0:4,:].sum().round(6) == A[:,0:4].sum().round(6), 'Directed networks are not supported'
    if k == n: return np.arange(0,n), A.sum()/2
    if k_lims[1] > n-k:
        print(':: WARNING: upper limit {} of the k_lims is above the available pool ' \
              ' of values.'.format(k_lims[1]))
        k_lims = (k_lims[0], np.min([k, n-k]))
        print('::      --> readjusted to {}.'.format(k_lims))
        
    find_maxima = ls_mode == 'best'
    w_thres = np.quantile(A[A > 0.0], 1-w_quantile)
    A_as = np.argsort(A)[:,::-1]
    A_beam = create_beam_array(A, A_as, w_thres)
    assert A_beam.shape[1] > 0, 'Set w_quantile is too small, A_beam has no elements'
    if verbose and A_beam.shape[1] < 10:
        print(':: WARNING: determined beam width is narrow @ {}, optimization result ' \
              'might suffer, for better result try increasing w_quantile'
              .format(A_beam.shape[1]))
    
    # Initialize
    t0 = process_time()
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
        H_w = __sub_sum(A, np.where(H)[0])
        Ho, Ho_w, ao, bo = ls_one_n_beam(H, H_w, A, A_beam, alpha=ao, beta=bo,
                                         tol=ls_tol, find_maxima=find_maxima,
                                         one_in_k=one_in_k, verbose=verbose)
    else:
        assert init_solution.shape[0] == n
        assert init_solution.sum() == k
        Ho = np.zeros(n, dtype=bool)
        Ho[init_solution] = True
        _, ao, bo = initialize_degree_vecs(A, H)    
    
    delta_t = process_time() - t0
    print(':: Initialization and first local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), Ho_w))
    
    i = i0 = 0
    stop = False
    while not stop:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and not stop:
            
            # 1. Perturbate
            if k_cur > 0:
                H, H_w, ap, bp = shake(A, Ho, k, k_cur, ao, bo, p_w, use_pref_attachment)
                assert k - (H & Ho).sum() == k_cur, f'k={k}, |H & Ho|={(H & Ho).sum()}, p={k_cur}'
                if verbose: print(':: Perturbation @ depth ', k_cur)
            else:
                H, H_w = Ho.copy(), Ho_w
                ap, bp = ao.copy(), bo.copy()
            
            # 2. Find local improvement
            H, H_w, ap, bp = ls_one_n_beam(H, H_w, A, A_beam, alpha=ap, beta=bp, 
                                           tol=ls_tol, find_maxima=find_maxima, 
                                           one_in_k=one_in_k, verbose=verbose)
            if verbose and find_maxima:
                if H_w != Ho_w: 
                    print(':: Local maxima:', Ho_w, '\n')
            i += 1
            i0 += 1
            update_cond = svns_score(H_w, Ho_w, H, Ho, k) > 1 + theta if svns else H_w > Ho_w
            if update_cond:
                delta_w = (H_w-Ho_w) / Ho_w * 100
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                print(':: iteration: {}, distance in iterations to earlier update: {}'
                      .format(i, i0))
                print(50*'--')
                i0 = 0
                Ho_w = H_w
                Ho = H
                ao = ap.copy()
                bo = bp.copy()
                k_cur = k_lims[0]
            else:
                k_cur += k_step
            stop = (i >= max_iter or i0 >= max_iter_upd or process_time() - t0 >= timetol)
                
    delta_t = str(td(seconds=(process_time()-t0)))
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f} ({:.6f} per node)'
          .format(delta_t, i, Ho_w, Ho_w / k))
    return np.where(Ho)[0], Ho_w

def OVNSfs(k: int, A: np.array, fss: list, k_lims: tuple, k_step: int=1, timetol: int=300,
           ls_tol: float=0.0, ls_mode: str='best', beta_ratio: float=0.5, seed: int=None, 
           w_quantile: float=0.01, init_mode='drop-initial', svns = False, theta: float=0.06, 
           one_in_k=False, max_iter: int=100000, max_iter_upd: int=100000, 
           init_solution: tuple=None, verbose=False): 
    """Variable neighborhood search heuristic for the set constrained variant of HkSP.
    
    Parameters
    ----------
    k : value of k in HkS (number of treatment groups)
    A : weighted adjacency matrix of the input network
    fss : set of pre-enumerated seed node configurations
    k_lims : range for search depth
    k_step : defines how many search depth steps to increment at once after failing update
    timetol : set the target time in seconds for the run
    ls_tol : set a tolerance for update (update is approved if improvement over current 
             best solution is at least ls_tol)
    ls_mode : define search mode, options are `best` and `first`, first being the default
    beta_ratio : sets the weights for the `weighted-deg` initialization strategy in range
                 [0.0, 1.0] where 0 gives all weight to degree ranking and 1 to weighted 
                 degree ranking.
    seed : seed for the random number generator (note: due to lack of numba support for numpy, 
           some numpy.random library calls don't allow seed input currently)
    w_quantile : define the quantile of edge weights that will be explored during each local 
                 neighorhood update iteration, lower values in increase the
    max_iter : maximum number of iterations after last successful update
    init_solution : initial solution as a tuple of np.arrays containing the node indices, 
                    where 1 index has the configuration of force selected seed nodes and 
                    0 has the remaining part of the solution
    
    Returns
    -------
    Ho : non-force selected solution graphlets as row indeces in the adjacency matrix
    Ho_fs : force selected solution graphlets as row indeces in the adjacency matrix
    Ho_w : value of the objective function for the solution
    """
    n = A.shape[0]
    assert ls_mode in ['first','best'], "Invalid local search mode; choose either " \
                                        " 'first' or 'best'" 
    assert k <= n, 'Input k value is greater than n; select k such that k <= n'
    assert 0.0 < w_quantile <= 1.0, 'Improper value, select value from range (0.0,1.0]'
    assert A[0:4,:].sum().round(6) == A[:,0:4].sum().round(6), 'Directed networks are ' \
                                                               'not supported'
    if k == n: return np.arange(0,n), A.sum()/2
    if k_lims[1] > n-k:
        print(':: WARNING: upper limit {} of the k_lims is above the available pool ' \
              'of values.'.format(k_lims[1]))
        k_lims = (k_lims[0], np.min([k, n-k]))
        print('::      --> readjusted to {}.'.format(k_lims))
    
    n_ss = __to_len_classes(fss)
    assert k >= min(n_ss.keys()), 'Input k value is below shortest available ' \
                                  'seed node configuration length'
    
    find_maxima = ls_mode == 'best'
    w_thres = np.quantile(A[A > 0.0], 1-w_quantile)
    A_as = np.argsort(A)[:,::-1]
    A_beam = create_beam_array(A, A_as, w_thres)
    assert A_beam.shape[1] > 0, 'Set w_quantile is too small, A_beam has no elements'
    if verbose and A_beam.shape[1] < 10:
        print(':: WARNING: determined beam width is narrow @ {}, optimization result ' \
              'might suffer, for better result try increasing w_quantile'
              .format(A_beam.shape[1]))
    
    # Initialize
    t0 = process_time()
    rsums = np.sum(A, axis=1)
    p_w = rsums / rsums.sum()
    if init_solution is None:
        if init_mode == 'drop-initial':
            H, Ho_fs, _, _ = init_solution_drop_initial_fs(A, k, fss)
        elif init_mode == 'weighted-deg':
            H, Ho_fs, _, = init_solution_weighted_degree_ranking_fs(A, k, fss, beta_ratio)
        else:
            H, Ho_fs = init_random_fs(A, k, fss)
        _, ao, bo = initialize_degree_vecs(A, H | Ho_fs)
        H_w = __sub_sum(A, np.where(H | Ho_fs)[0])
        Ho, Ho_w, ao, bo = ls_one_n_beam_fs(H, Ho_fs, H_w, A, A_beam, ao, bo, 
                                            tol=ls_tol, find_maxima=find_maxima,
                                            one_in_k=one_in_k, verbose=verbose)
    else:
        Ho = np.zeros(n, dtype=bool)
        Ho_fs = np.zeros(n, dtype=bool)
        Ho[init_solution[0]] = True
        Ho_fs[init_solution[1]] = True
        assert (Ho | Ho_fs).sum() == k
        assert (Ho & Ho_fs).sum() == 0
        Hu = np.where(Ho | Ho_fs)[0]
        H_w = Ho_w = __sub_sum(A, Hu)

    delta_t = process_time()-t0
    print(':: Initial local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), Ho_w))
    
    i = i0 = 0
    stop = False
    while not stop:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and not stop:
            
            # 1. Perturbate    
            if k_cur > 0:
                H, H_fs, ap, bp = shake_fs(A, Ho, Ho_fs, fss, n_ss, k, k_cur, ao, bo)
                if verbose: print(':: Perturbation @ depth ', k_cur)
            else:
                H, H_fs, H_w = Ho.copy(), Ho_fs.copy(), Ho_w
                ap, bp = ao.copy(), bo.copy()
                
            # 2. Find local improvement
            H_w = __sub_sum(A, np.where(H | H_fs)[0])
            H, H_w, ap, bp = ls_one_n_beam_fs(H, H_fs, H_w, A, A_beam, ap, bp, 
                                              tol=ls_tol, find_maxima=find_maxima,
                                              one_in_k=one_in_k, verbose=verbose)
            if verbose and find_maxima:
                if H_w != Ho_w:
                    print(':: Local maxima:', Ho_w, '\n')
            i += 1
            i0 += 1
            update_cond = svns_score(H_w, Ho_w,
                                     H | H_fs, Ho | Ho_fs,
                                     k) > 1 + theta if svns else H_w > Ho_w
            if update_cond:
                delta_w = (H_w-Ho_w) / Ho_w * 100 
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                print(':: iteration: {}, distance in iterations to earlier update: {}'.format(i, i0))
                print(50*'--')
                i0 = 0
                Ho_w = H_w
                Ho = H
                Ho_fs = H_fs
                ao = ap.copy()
                bo = bp.copy()
                k_cur = k_lims[0]
            else:
                k_cur += k_step
            stop = (i >= max_iter or i0 >= max_iter_upd or process_time() - t0 >= timetol)
                
    delta_t = str(td(seconds=(process_time()-t0)))
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f} ({:.6f} per node)'
          .format(delta_t, i, Ho_w, Ho_w / k))
    return np.where(Ho)[0], np.where(Ho_fs)[0], Ho_w

@njit
def local_search_bvns(Ho, Ho_w, A, verbose=False, tol=0.0, ls_tol=0.0, 
                      find_maxima=False):
    """Computes local search in the 1-neighborhood of the Ho set. Select between
    'first improvement' and 'best improvement' strategies with find_maxima parameter.
    (best improvement strategy finds the local maxima).
    """
    H_t0_w = 0.0
    n = A.shape[0]

    while Ho_w - H_t0_w > tol * Ho_w:
        H_t0_w = Ho_w
        Ho, Ho_w = ls_one_n(Ho, Ho_w, A, verbose=verbose, tol=ls_tol)
        if not find_maxima:
            return Ho, Ho_w

    if verbose: print(':: Local maxima:', Ho_w, '\n')
    return Ho, Ho_w

def BVNS(k: int, A: np.array, k_lims: tuple, k_step: int=1, timetol: int=300, 
        ls_tol: float=0.0, ls_mode='first', init_mode='random', seed: int=None, 
        max_iter: int=100000, verbose=False): 
    """Simple variable neighborhood search heuristic (Brimberg et al. 2009) for the HkS problem. 
    Uses 'first improvement' mode by default, while 'best improvement' will search until local 
    maxima is found.
    """
    
    assert ls_mode in ['first','best'], "Invalid local search mode; choose either 'first' or 'best'" 
    assert init_mode in ['heaviest-edge','weighted-deg','random']
    n = A.shape[0]
    assert k <= n, 'Input k value is greater than n; select k such that k <= n'
    assert A[0,:].sum() == A[:,0].sum(), 'Directed networks are not supported'
    if k == n: return np.arange(0,n), A.sum()/2
    if k_lims[1] > n-k:
        print(':: WARNING: upper limit {} of the k_lims is above the available pool of values.'.format(k_lims[1]))
        k_lims = (k_lims[0], np.min([k, n-k]))
        print('::      --> readjusted to {}.'.format(k_lims))
        
    find_maxima = ls_mode == 'best'
    use_pref_attachment = False
    ksum = np.sum(np.triu(A)>0)
    p_w = np.sum(A, axis=0)
    
    # Initialize
    t0 = process_time()
    H = np.random.choice(np.arange(n), k, replace=False)
    H_w = __sub_sum(A, H)
    Ho, Ho_w = local_search_bvns(H, H_w, A, verbose=verbose, 
                                         find_maxima=find_maxima, 
                                         tol=ls_tol)
    delta_t = process_time()-t0
    print(':: Initial local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), Ho_w))
    
    i = i0 = 0
    while process_time() - t0 < timetol:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and process_time() - t0 < timetol:
            # Perturbate
            H = shake(A, Ho, k, k_cur, p_w, use_pref_attachment)
            if verbose: print(':: Perturbation @ depth ', k_cur)
            # Find local improvement
            H_w = __sub_sum(A, H)
            H, H_w = local_search_bvns(H, H_w, A, verbose=verbose,
                                       find_maxima=find_maxima, 
                                       tol=ls_tol)
            i += 1
            if H_w > Ho_w:
                delta_w = (H_w-Ho_w) / Ho_w * 100 
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                if i0 is not None:
                    print(':: iteration: {}, iterative distance to earlier maxima: {}'.format(i, i-i0))
                print(50*'--')
                i0 = i
                Ho_w = H_w
                Ho = H
                k_cur = k_lims[0]
            else:
                k_cur += k_step
                
    delta_t = str(td(seconds=(process_time()-t0)))
    print(':: Run completed @ {}, final f value: {:.6f}'.format(delta_t, Ho_w))
    return Ho, Ho_w
