from time import process_time
from datetime import timedelta as td
import numpy as np
from numba import *
from kmmi.heuristics.initialize import *

@njit
def create_beam_array(A, A_as, w_thres):
    """Compute a beam array out of adjacency matrix A. In a beam array each row 
    i will contain the indexes of all connected nodes for node i in sorted order  
    based on the link weight."""
    # A as arg-sorted nonzero, reversed
    n = A.shape[0]
    A_beam = np.zeros((n,n), dtype=np.int32) - 1 
    maxlen = 0
    for i in range(n):
        j = 0
        for k in A_as[i,:]:
            if A[i,k] > w_thres:
                A_beam[i,j] = k
                j+=1
            else:
                if j > maxlen:
                    maxlen = j
                break
    
    return A_beam[:,:maxlen]

@njit
def __sub_sum(A, u):
    eps = 0
    for ui in u:
        for uj in u:
            eps += A[ui,uj]
    return eps / 2

@njit
def ls_one_n_beam(Ho, Ho_w, A, A_beam, tol=0.0, verbose=False):
    """Computes local search in the 1-neighborhood of the Ho set such that 
    node is selected using beam criterion; the space of neighbors with heaviest 
    links in the 1-neighborhood of the current H nodes is searched first. First 
    improvement of the objective function is returned. Tol sets the allowed 
    fraction of improvement over earlier best value of objective function.
    """
    n = A_beam.shape[0]
    k = len(Ho)
    np.random.shuffle(Ho)
    
    L = 0
    for j in range(n):
        for i in range(k):
            v = Ho[i]
            u = A_beam[v,j]
            if u < 0: 
                continue
            L += 1
            if verbose:
                if L % 10000 == 0:
                    print(':: Iterations: ', L)
            if u not in Ho:
                v = Ho[i]
                L += 1
                H = np.array([vi if vi != v else u for vi in Ho])
                H_w = __sub_sum(A, H)
                if H_w > Ho_w + tol * Ho_w:
                    if verbose:
                        print(':: Improvement found: +', (H_w-Ho_w))
                        print(':: Objective function value: ', Ho_w,', iters: ', L)
                    return H, H_w

    if verbose: print(':: No improvement found during local search.')
    return Ho, Ho_w

@njit
def ls_one_n_beam_fs(Ho, Ho_fs, Ho_w, A, A_beam, tol=0.0, verbose=False):
    """Computes local search in the 1-neighborhood of the Ho set such that 
    node is selected using beam criterion; the space of neighbors with heaviest 
    links in the 1-neighborhood of the current H nodes is searched first. First 
    improvement of the objective function is returned. tol sets the allowed 
    fraction of improvement over earlier best value of objective function.
    """
    n = A_beam.shape[1]
    k1, k2 = len(Ho), len(Ho_fs)
    np.random.shuffle(Ho)
    
    L = 0                        
    for j in range(n):
        for i in range(k1+k2):
            v = Ho[i] if i < k1 else Ho_fs[i-k1]
            u = A_beam[v,j]
            if u < 0: continue
            if u not in Ho and u not in Ho_fs:
                for q in Ho:
                    L += 1
                    H = [vi if vi != q else u for vi in Ho]
                    H_f = H.copy()
                    for h in Ho_fs:
                        H_f.append(h)
                    H_f = np.array(H_f)
                    H_w = __sub_sum(A, H_f)
                    if H_w > Ho_w + tol * Ho_w:
                        if verbose:
                            print(':: Improvement found: +', (H_w-Ho_w))
                            print(':: Objective function value: ', Ho_w,', iters: ', L)
                        return np.array(H), H_w

    if verbose: print(':: No improvement found during local search.')
    return Ho, Ho_w

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
    
@njit
def local_search_ovns(Ho, Ho_w, A, A_beam, verbose=False, tol=0.0, ls_tol=0.0, 
                      find_maxima=False):
    """Computes local search in the 1-neighborhood of the Ho set. Select between
    'first improvement' and 'best improvement' strategies with find_maxima parameter.
    (best improvement strategy finds the local maxima).
    """
    H_t0_w = 0.0
    n = A.shape[0]

    while Ho_w - H_t0_w > tol * Ho_w:
        H_t0_w = Ho_w
        Ho, Ho_w = ls_one_n_beam(Ho, Ho_w, A, A_beam, ls_tol, verbose)
        if not find_maxima:
            return Ho, Ho_w

    if verbose: print(':: Local maxima:', Ho_w, '\n')
    return Ho, Ho_w

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

def shake(A: np.array, Ho: np.array, k: int, p: int, p_w: np.array=None, 
          use_pref_attachment=False):
    """Implements the perturbation routine for the VNS (variable neighborhood search)
    H by randomly drawing p node ids from the H without replacement and replacing them 
    with p randomly drawn node ids from the complement of H.
    
    Parameter use_pref_attachment controls for the weighting of the random distribution. 
    If True allows for preferential weighting based on weighted node degree. Default, 
    False is uniformly distributed.
    """
    n = A.shape[0]
    replace_idxs = np.random.choice(k, size=p, replace=False)
    
    H = Ho.copy()
    for i in range(p):
        while True:
            if use_pref_attachment:
                H_prime = np.random.choice(n, size=1, replace=False, p=p_w)
            else:
                H_prime = np.random.randint(0, A.shape[0])
            if H_prime not in H:
                H[replace_idxs[i]] = H_prime
                break
    return H

def shake_fs(A: np.array, Ho: np.array, H_fs_opt: np.array, fss: list, n_ss: dict, k: int, p: int):
    """Implements the perturbation routine for the VNS when forced selection is used.
    
        Method has two different behaviors that are dependent on the p value and H_fs size:
        1. when p is smaller than the size of H_fs, only nodes in H are swapped
        2. when p is at least the size of H_{fs}, then also the H_fs configuration 
           is swapped
    """
    assert k > len(H_fs_opt), 'size of H_fs is larger than k, try increasing k'
    n = A.shape[0]
    Aset = set(np.arange(A.shape[0]))
        
    if p <= len(Ho):
        H = Ho.copy()
        H_fs = H_fs_opt.copy()
        idxs = np.random.choice(len(Ho), size=p, replace=False)
        HC = Aset - (set(Ho) | set(H_fs_opt))
        H[idxs] = np.random.choice(list(HC), size=p, replace=False)
    else: 
        H_fs = fss[__sample_up_to_class_p(k-1, n_ss)]
        p_diff = p - len(H_fs)
        l_diff = k - len(H_fs)
        H = np.array(list(set(Ho) - set(H_fs)))
        if p_diff > 0:
            HC = Aset - (set(H) | set(H_fs))
            Hn = np.random.choice(list(HC), size=p_diff, replace=False)
            l_diff = k-len(H_fs)-len(Hn)
        if len(H) >= l_diff:
            H = np.random.choice(list(H), size=l_diff, replace=False)
        if p_diff > 0:
            H = np.concatenate([H,Hn], axis=0)
        if len(H) + len(H_fs) < k:
            HC = Aset - (set(H) | set(H_fs))
            Hn = np.random.choice(list(HC), size=k-len(H)-len(H_fs), replace=False)
            H = np.concatenate([H,Hn], axis=0)
            
    Hfs_len = len(H) + len(H_fs)
    assert Hfs_len == k, f'H combined size {Hfs_len} does not match with k={k}'
    assert len(set(H_fs) & set(H)) == 0, f'There is overlap in the selection\n * : ' \
                                          '{set(H_fs) & set(H)}'
    return H, np.array(H_fs) 

@njit
def sample_from_HC(A, H):
    while True:
        H_prime = np.random.randint(0, A.shape[0])
        if H_prime not in H:
            return H_prime
        
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
         ls_mode='first', use_pref_attachment=False, init_mode='heaviest-edge', beta_ratio: float=0.5, 
         seed: int=None, max_iter: int=100000, w_quantile=0.01, verbose=False): 
    """Variable neighborhood search heuristic for the HkS problem. Uses 'first improvement' 
    mode by default, while 'best improvement' will search until local maxima is found.
    
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
    seed : seed for the RNG
    w_quantile : define the quantile of edge weights that will be explored during each local 
                 neighorhood update iteration, lower values in increase the
    max_iter : maximum number of iterations after last successful update
    
    Returns
    -------
    Ho : selected solution graphlets as row indeces in the adjacency matrix
    Ho_w : value of the objective function for the solution
    """
    
    assert ls_mode in ['first','best'], "Invalid local search mode; choose either 'first' or 'best'" 
    assert init_mode in ['heaviest-edge','weighted-deg','random']
    assert 0.0 < w_quantile < 1.0, 'Improper value, select value from range [0.0,1.0]'
    n = A.shape[0]
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

    # Initialize
    t0 = process_time()
    p_w = None
    if init_mode == 'heaviest-edge':
        H, H_w = init_solution_heaviest_edge_ranking(A, k)
    elif init_mode == 'weighted-deg':
        H, H_w, p_w = init_solution_weighted_degree_ranking(A, k, beta_ratio=beta_ratio)
    else:
        H = np.random.choice(np.arange(n), k, replace=False)
        H_w = __sub_sum(A, H)
        
    Ho, Ho_w = local_search_ovns(H, H_w, A, A_beam, verbose=verbose,
                                    find_maxima=find_maxima, tol=ls_tol, ls_tol=ls_tol)
    delta_t = process_time()-t0
    print(':: Initial local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), Ho_w))
    
    i = i0 = 0
    while process_time() - t0 < timetol and i0 < max_iter:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and process_time() - t0 < timetol:
            # Perturbate
            H = shake(A, Ho, k, k_cur, p_w, use_pref_attachment)
            if verbose: print(':: Perturbation @ depth ', k_cur)
            # Find local improvement
            H_w = __sub_sum(A, H)
            H, H_w = local_search_ovns(H, H_w, A, A_beam, verbose=verbose, 
                                       find_maxima=find_maxima, tol=ls_tol, ls_tol=ls_tol)
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
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f}'.format(delta_t, i, Ho_w))
    return Ho, Ho_w

def OVNSfs(k: int, A: np.array, fss: list, k_lims: tuple, k_step: int=1, timetol: int=300,
           ls_tol: float=0.0, ls_mode: str='best', beta_ratio: float=0.25, seed: int=None, 
           w_quantile: float=0.01, max_iter: int=100000, init_solution: tuple=None, verbose=False): 
    """Variable neighborhood search heuristic for the HkS problem - augmented with forced node 
    selection for pre-defined set of nodes. 
    
    'First improvement' mode is used by default, while 'best improvement' will search until 
    local maxima is found.
    
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
    seed : seed for the RNG
    w_quantile : define the quantile of edge weights that will be explored during each local 
                 neighorhood update iteration, lower values in increase the
    max_iter : maximum number of iterations after last successful update
    init_solution : initial solution as a tuple of np.arrays, where 1 index has the 
                    configuration of force selected seed nodes and 0 has the remaining
                    part of the solution
    
    Returns
    -------
    Ho : non-force selected solution graphlets as row indeces in the adjacency matrix
    Ho_fs : force selected solution graphlets as row indeces in the adjacency matrix
    Ho_w : value of the objective function for the solution
    """
    
    assert ls_mode in ['first','best'], "Invalid local search mode; choose either " \
                                        " 'first' or 'best'" 
    n = A.shape[0]
    assert k <= n, 'Input k value is greater than n; select k such that k <= n'
    assert A[0:4,:].sum().round(6) == A[:,0:4].sum().round(6), 'Directed networks are ' \
                                                               'not supported'
    assert 0.0 < w_quantile < 1.0, 'Improper value, select value from range [0.0,1.0]'

    if k == n: return np.arange(0,n), A.sum()/2
    fss_max = len(max(fss, key=len))
    if k_lims[1] > n-k:
        print(':: WARNING: upper limit {} of the k_lims is above the available pool ' \
              'of values.'.format(k_lims[1]))
        k_lims = (k_lims[0], np.min([k, n-k]))
        print('::      --> readjusted to {}.'.format(k_lims))
    
    n_ss = __to_len_classes(fss)
    assert k >= min(n_ss.keys()), 'Input k value is smaller than the lowest available ' \
                                  'seed node configuration'
    
    find_maxima = ls_mode == 'best'
    w_thres = np.quantile(A[A > 0.0], 1-w_quantile)
    A_as = np.argsort(A)[:,::-1]
    A_beam = create_beam_array(A, A_as, w_thres)
    
    # Initialize
    t0 = process_time()
    p_w = None
    if init_solution is None:
        H, H_w, p_w, Ho_fs = init_solution_weighted_degree_ranking_fs(A, k, fss, beta_ratio)
        Ho, Ho_w = local_search_ovns_fs(H, Ho_fs, H_w, A, A_beam=A_beam, verbose=verbose,
                                        find_maxima=find_maxima, tol=ls_tol, ls_tol=ls_tol)
    else:
        H, H_fs = init_solution
        Ho = H
        Ho_fs = H_fs
        H_f = np.concatenate([H, H_fs], axis=0)
        H_w = Ho_w = __sub_sum(A, H_f)
            
    delta_t = process_time()-t0
    print(':: Initial local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), Ho_w))
    
    i = i0 = 0
    while process_time() - t0 < timetol and i0 < max_iter:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and process_time() - t0 < timetol:
            # Perturbate    
            H, H_fs = shake_fs(A, Ho, Ho_fs, fss, n_ss, k, k_cur)
            if verbose: print(':: Perturbation @ depth ', k_cur)
            
            # Find local improvement
            H_f = np.concatenate([H, H_fs], axis=0)
            H_w = __sub_sum(A, H_f)
            H, H_w = local_search_ovns_fs(H, H_fs, H_w, A, A_beam, verbose=verbose, 
                                       find_maxima=find_maxima, tol=ls_tol, ls_tol=ls_tol)
            i += 1
            if H_w > Ho_w:
                delta_w = (H_w-Ho_w) / Ho_w * 100 
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                if i0 is not None:
                    print(':: iteration: {}, distance in iterations to earlier maxima: {}'.format(i, i-i0))
                print(50*'--')
                i0 = i
                Ho_w = H_w
                Ho = H
                Ho_fs = H_fs
                k_cur = k_lims[0]
            else:
                k_cur += k_step
                
    delta_t = str(td(seconds=(process_time()-t0)))
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f}'.format(delta_t, i, Ho_w))
    return Ho, Ho_fs, Ho_w

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
    p_w = None
    
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
    while process_time() - t0 < timetol and i0 < max_iter:
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
