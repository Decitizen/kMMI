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
def ls_one_n_beam(H_opt, H_opt_w, A, A_beam, tol=0.0, verbose=False):
    """Computes local search in the 1-neighborhood of the H_opt set such that 
    node is selected using beam criterion; the space of neighbors with heaviest 
    links in the 1-neighborhood of the current H nodes is searched first. First 
    improvement of the objective function is returned. Tol sets the allowed 
    fraction of improvement over earlier best value of objective function.
    """
    n = A_beam.shape[0]
    k = len(H_opt)
    np.random.shuffle(H_opt)
    
    L = 0
    for j in range(n):
        for i in range(k):
            v = H_opt[i]
            u = A_beam[v,j]
            if u < 0: 
                continue
            L += 1
            if verbose:
                if L % 10000 == 0:
                    print(':: Iterations: ', L)
            if u not in H_opt:
                v = H_opt[i]
                L += 1
                H = np.array([vi if vi != v else u for vi in H_opt])
                H_w = A[H,:][:,H].sum() / 2
                if H_w > H_opt_w + tol * H_opt_w:
                    if verbose:
                        print(':: Improvement found: +', (H_w-H_opt_w))
                        print(':: Objective function value: ', H_opt_w,', iters: ', L)
                    return H, H_w

    if verbose: print(':: No improvement found during local search.')
    return H_opt, H_opt_w

@njit
def ls_one_n_beam_fs(H_opt, H_opt_fs, H_opt_w, A, A_beam, tol=0.0, verbose=False):
    """Computes local search in the 1-neighborhood of the H_opt set such that 
    node is selected using beam criterion; the space of neighbors with heaviest 
    links in the 1-neighborhood of the current H nodes is searched first. First 
    improvement of the objective function is returned. Tol sets the allowed 
    fraction of improvement over earlier best value of objective function.
    """
    n = A_beam.shape[1]
    k1, k2 = len(H_opt), len(H_opt_fs)
    np.random.shuffle(H_opt)
    
    L = 0                        
    for j in range(n):
        for i in range(k1+k2):
            v = H_opt[i] if i < k1 else H_opt_fs[i-k1]
            u = A_beam[v,j]
            if u < 0: continue
            if u not in H_opt and u not in H_opt_fs:
                for q in H_opt:
                    L += 1
                    H = [vi if vi != q else u for vi in H_opt]
                    H_f = H.copy()
                    for h in H_opt_fs:
                        H_f.append(h)
                    H_f = np.array(H_f)
                    H_w = A[H_f,:][:,H_f].sum() / 2
                    if H_w > H_opt_w + tol * H_opt_w:
                        if verbose:
                            print(':: Improvement found: +', (H_w-H_opt_w))
                            print(':: Objective function value: ', H_opt_w,', iters: ', L)
                        return np.array(H), H_w

    if verbose: print(':: No improvement found during local search.')
    return H_opt, H_opt_w

@njit
def ls_one_n(H_opt, H_opt_w, A, tol=0.0, verbose=False):
    """Computes local search in the 1-neighborhood of the H_opt set in
    random ordering.
    """
    n = A.shape[0]
    k = len(H_opt)
    Hc = np.array([v for v in range(n) if v not in H_opt])
    np.random.shuffle(Hc)
    np.random.shuffle(H_opt)
    
    L = 1
    for u in Hc:
        for v in H_opt:
            H = np.array([vi if vi != v else u for vi in H_opt])
            H_w = A[H,:][:,H].sum() / 2
            L += 1
            if verbose:
                if L % 10000 == 0:
                    print(':: Iterations: ', L)
  
            if H_w > H_opt_w + tol*H_opt_w:
                if verbose:
                    print(':: Improvement found: +', (H_w-H_opt_w))
                    print(':: Objective function value: ', H_opt_w,', iters: ', L)
                return H, H_w
            
    if verbose: print(':: No improvement found during local search.')
    return H_opt, H_opt_w
    
@njit
def local_search_ovns(H_opt, H_opt_w, A, A_beam, verbose=False, tol=0.0, ls_tol=0.0, 
                      find_maxima=False):
    """Computes local search in the 1-neighborhood of the H_opt set. Select between
    'first improvement' and 'best improvement' strategies with find_maxima parameter.
    (best improvement strategy finds the local maxima).
    """
    H_t0_w = 0.0
    n = A.shape[0]

    while H_opt_w - H_t0_w > tol * H_opt_w:
        H_t0_w = H_opt_w
        H_opt, H_opt_w = ls_one_n_beam(H_opt, H_opt_w, A, A_beam, 
                                       verbose=verbose, tol=ls_tol)
        if not find_maxima:
            return H_opt, H_opt_w

    if verbose: print(':: Local maxima:', H_opt_w, '\n')
    return H_opt, H_opt_w

@njit
def local_search_ovns_fs(H_opt, H_opt_fs, H_opt_w, A, A_beam, verbose=False, 
                      tol=0.0, ls_tol=0.0, find_maxima=False):
    """Computes local search in the 1-neighborhood of the H_opt set. Select between
    'first improvement' and 'best improvement' strategies with find_maxima parameter.
    (best improvement strategy finds the local maxima).
    """
    H_t0_w = 0.0
    n = A.shape[0]
    
    while H_opt_w - H_t0_w > tol * H_opt_w:
        H_t0_w = H_opt_w
        H_opt, H_opt_w = ls_one_n_beam_fs(H_opt, 
                                          H_opt_fs, 
                                          H_opt_w, 
                                          A, A_beam, 
                                          verbose=verbose, 
                                          tol=ls_tol)
        if not find_maxima:
            return H_opt, H_opt_w

    if verbose: print(':: Local maxima:', H_opt_w, '\n')
    return H_opt, H_opt_w

def shake(A: np.array, H_opt: np.array, k: int, p: int, p_w: np.array=None, 
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
    
    H = H_opt.copy()
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

def shake_fs(A: np.array, H_opt: np.array, H_fs_opt: np.array, fss: list, k: int, p: int):
    """Implements the perturbation routine for the VNS when forced selection is used 
    H by randomly drawing p node ids from the H without replacement and replacing them 
    with p randomly drawn node ids from the complement of H.
    """
    n = A.shape[0]
    H = H_opt.copy()
    H_fs = fss[np.random.choice(len(fss))]
    
    l_diff = len(H_fs_opt) - len(H_fs)
    if l_diff != 0:
        if l_diff > 0: # Add l_diff more
            Hp = [sample_from_HC(A, H) for _ in range(l_diff)]
            Hp += list(H)
            H = np.array(Hp)
        
        else: # Remove
            idxs = np.random.choice(len(H_opt), size=-l_diff, replace=False)            
            H = np.array([v for i,v in enumerate(H) if i not in idxs])
    
    if p > len(H_fs):
        replace_idxs = np.random.choice(len(H), size=p-len(H_fs), replace=False)
        for i in range(p - len(H_fs)):
            HC = np.concatenate([H,H_fs], axis=0)
            H[replace_idxs[i]] = sample_from_HC(A, HC)
            break
    
    Hfs_len = len(H) + len(H_fs)
    assert Hfs_len == k, f'H combined size {Hfs_len} does not match with k={k}'
    return H, np.array(H_fs) 

@njit
def sample_from_HC(A, H):
    while True:
        H_prime = np.random.randint(0, A.shape[0])
        if H_prime not in H:
            return H_prime
        
def OVNS(k: int, A: np.array, k_lims: tuple, k_step: int=1, timetol: int=300, 
        ls_tol: float=0.0, ls_mode='first', use_pref_attachment=False, init_mode='heaviest-edge', 
        beta_ratio: float=0.5, seed: int=None, max_iter: int=100000, w_quantile=0.99, verbose=False): 
    """Variable neighborhood search heuristic for the HkS problem. Uses 'first improvement' 
    mode by default, while 'best improvement' will search until local maxima is found.
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
    w_thres = np.quantile(A[A > 0.0], w_quantile)
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
        H_w = A[H,:][:,H].sum() / 2

    H_gopt, H_gopt_w = local_search_ovns(H, H_w, A, A_beam, verbose=verbose,
                                    find_maxima=find_maxima, tol=ls_tol, ls_tol=ls_tol)
    delta_t = process_time()-t0
    print(':: Initial local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), H_gopt_w))
    
    i = i0 = 0
    while process_time() - t0 < timetol and i0 < max_iter:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and process_time() - t0 < timetol:
            # Perturbate
            H = shake(A, H_gopt, k, k_cur, p_w, use_pref_attachment)
            if verbose: print(':: Perturbation @ depth ', k_cur)
            # Find local improvement
            H_w = np.sum(A[H,:][:,H]) / 2
            H, H_w = local_search_ovns(H, H_w, A, A_beam, verbose=verbose, 
                                       find_maxima=find_maxima, tol=ls_tol, ls_tol=ls_tol)
            i += 1
            if H_w > H_gopt_w:
                delta_w = (H_w-H_gopt_w) / H_gopt_w * 100 
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                if i0 is not None:
                    print(':: iteration: {}, iterative distance to earlier maxima: {}'.format(i, i-i0))
                print(50*'--')
                i0 = i
                H_gopt_w = H_w
                H_gopt = H
                k_cur = k_lims[0]
            else:
                k_cur += k_step
                
    delta_t = str(td(seconds=(process_time()-t0)))
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f}'.format(delta_t, i, H_opt_w))
    return H_gopt, H_gopt_w

def OVNSfs(k: int, A: np.array, fss: list, k_lims: tuple, k_step: int=1, timetol: int=300,
            ls_tol: float=0.0, ls_mode: str='best', beta_ratio: float=0.25, seed: int=None, 
            w_quantile: float=0.99, max_iter: int=100000, verbose=False): 
    """Variable neighborhood search heuristic for the HkS problem. Uses 'first improvement' 
    mode by default, while 'best improvement' will search until local maxima is found.
    """
    
    assert ls_mode in ['first','best'], "Invalid local search mode; choose either 'first' or 'best'" 
    n = A.shape[0]
    assert k <= n, 'Input k value is greater than n; select k such that k <= n'
    assert A[0,:].sum() == A[:,0].sum(), 'Directed networks are not supported'
    if k == n: return np.arange(0,n), A.sum()/2
    fss_max = len(max(fss, key=len))
    if k_lims[1] > n-k:
        print(':: WARNING: upper limit {} of the k_lims is above the available pool of values.'.format(k_lims[1]))
        k_lims = (k_lims[0], np.min([k, n-k]))
        print('::      --> readjusted to {}.'.format(k_lims))
    
    find_maxima = ls_mode == 'best'
    w_thres = np.quantile(A[A > 0.0], w_quantile)
    A_as = np.argsort(A)[:,::-1]
    A_beam = create_beam_array(A, A_as, w_thres)
    
    # Initialize
    t0 = process_time()
    p_w = None
    H, H_w, p_w, H_opt_fs = init_solution_weighted_degree_ranking_fs(A, k, fss, beta_ratio)
    H_opt, H_opt_w = local_search_ovns_fs(H, H_opt_fs, H_w, A, A_beam=A_beam, verbose=verbose,
                                       find_maxima=find_maxima, tol=ls_tol, ls_tol=ls_tol)
    delta_t = process_time()-t0
    print(':: Initial local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), H_opt_w))
    
    i = i0 = 0
    while process_time() - t0 < timetol and i0 < max_iter:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and process_time() - t0 < timetol:
            # Perturbate    
            H, H_fs = shake_fs(A, H_opt, H_opt_fs, fss, k, k_cur)
            if verbose: print(':: Perturbation @ depth ', k_cur)
            
            # Find local improvement
            H_f = np.concatenate([H, H_fs], axis=0)
            H_w = np.sum(A[H_f,:][:,H_f]) / 2
            H, H_w = local_search_ovns_fs(H, H_fs, H_w, A, A_beam, verbose=verbose, 
                                       find_maxima=find_maxima, tol=ls_tol, ls_tol=ls_tol)
            i += 1
            if H_w > H_opt_w:
                delta_w = (H_w-H_opt_w) / H_opt_w * 100 
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                if i0 is not None:
                    print(':: iteration: {}, distance in iterations to earlier maxima: {}'.format(i, i-i0))
                print(50*'--')
                i0 = i
                H_opt_w = H_w
                H_opt = H
                H_opt_fs = H_fs
                k_cur = k_lims[0]
            else:
                k_cur += k_step
                
    delta_t = str(td(seconds=(process_time()-t0)))
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f}'.format(delta_t, i, H_opt_w))
    return H_opt, H_opt_fs, H_opt_w

@njit
def local_search_bvns(H_opt, H_opt_w, A, verbose=False, tol=0.0, ls_tol=0.0, 
                      find_maxima=False):
    """Computes local search in the 1-neighborhood of the H_opt set. Select between
    'first improvement' and 'best improvement' strategies with find_maxima parameter.
    (best improvement strategy finds the local maxima).
    """
    H_t0_w = 0.0
    n = A.shape[0]

    while H_opt_w - H_t0_w > tol * H_opt_w:
        H_t0_w = H_opt_w
        H_opt, H_opt_w = ls_one_n(H_opt, H_opt_w, A, verbose=verbose, tol=ls_tol)
        if not find_maxima:
            return H_opt, H_opt_w

    if verbose: print(':: Local maxima:', H_opt_w, '\n')
    return H_opt, H_opt_w

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
    H_w = A[H,:][:,H].sum() / 2
    H_gopt, H_gopt_w = local_search_bvns(H, H_w, A, verbose=verbose, 
                                         find_maxima=find_maxima, 
                                         tol=ls_tol)
    delta_t = process_time()-t0
    print(':: Initial local search completed.')
    print(':: LS took {}, value: {}\n'.format(str(td(seconds=delta_t)), H_gopt_w))
    
    i = i0 = 0
    while process_time() - t0 < timetol and i0 < max_iter:
        k_cur = k_lims[0]
        while k_cur <= k_lims[1] and process_time() - t0 < timetol:
            # Perturbate
            H = shake(A, H_gopt, k, k_cur, p_w, use_pref_attachment)
            if verbose: print(':: Perturbation @ depth ', k_cur)
            # Find local improvement
            H_w = np.sum(A[H,:][:,H]) / 2
            H, H_w = local_search_bvns(H, H_w, A, verbose=verbose,
                                       find_maxima=find_maxima, 
                                       tol=ls_tol)
            i += 1
            if H_w > H_gopt_w:
                delta_w = (H_w-H_gopt_w) / H_gopt_w * 100 
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                if i0 is not None:
                    print(':: iteration: {}, iterative distance to earlier maxima: {}'.format(i, i-i0))
                print(50*'--')
                i0 = i
                H_gopt_w = H_w
                H_gopt = H
                k_cur = k_lims[0]
            else:
                k_cur += k_step
                
    delta_t = str(td(seconds=(process_time()-t0)))
    print(':: Run completed @ {}, final f value: {:.6f}'.format(delta_t, H_gopt_w))
    return H_gopt, H_gopt_w
