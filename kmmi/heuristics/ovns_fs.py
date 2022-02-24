from time import process_time
from datetime import timedelta as td
import numpy as np
from numba import *
from kmmi.utils.utils import sub_sum
from kmmi.heuristics.initialize import *
from kmmi.heuristics.neighborhood_change import *
from kmmi.heuristics.neighborhood_search import *
from kmmi.heuristics.utils import __create_beam_array, __svns_score, __to_len_classes

def OVNS_fs(k: int, A: np.array, fss: list, k_lims: tuple, k_step: int=1, timetol: int=300,
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
    max_iter : defines the absolute upper bound for the number of executed iterations 
               after which the execution will be terminated (default: None)
    max_iter_upd : define the maximum number of iterations after last successful update
                   (default: 1e5)
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
    assert k != n, 'Input k value equals n, solution contains all nodes in the network'
    assert k < n, 'Input k value is greater than n; select k such that k < n'
    assert 0.0 < w_quantile <= 1.0, 'Improper value, select value from range (0.0,1.0]'
    assert A[0:4,:].sum().round(6) == A[:,0:4].sum().round(6), 'Directed networks are ' \
                                                               'not supported'
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
    A_beam = __create_beam_array(A, A_as, w_thres)
    assert A_beam.shape[1] > 0, 'Set w_quantile is too small, A_beam has no elements'
    if verbose and A_beam.shape[1] < 10:
        print(':: WARNING: determined beam width is narrow @ {}, optimization result ' \
              'might suffer, for better result try increasing w_quantile'
              .format(A_beam.shape[1]))
    
    hss = []
    hfs = []
    run_trace = []

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
        H_w = sub_sum(A, np.where(H | Ho_fs)[0])
        Ho, Ho_w, ao, bo = ls_one_n_beam_fs(H, Ho_fs, H_w, A, A_beam, ao, bo, 
                                            tol=ls_tol, find_maxima=find_maxima,
                                            one_in_k=one_in_k, verbose=verbose)
        hss.append(Ho)
        hfs.append(Ho_fs)
        run_trace.append((H_w, 0))      
    else:
        Ho = np.zeros(n, dtype=bool)
        Ho_fs = np.zeros(n, dtype=bool)
        Ho[init_solution[0]] = True
        Ho_fs[init_solution[1]] = True
        assert (Ho | Ho_fs).sum() == k
        assert (Ho & Ho_fs).sum() == 0
        Hu = np.where(Ho | Ho_fs)[0]
        H_w = Ho_w = sub_sum(A, Hu)

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
            H_w = sub_sum(A, np.where(H | H_fs)[0])
            H, H_w, ap, bp = ls_one_n_beam_fs(H, H_fs, H_w, A, A_beam, ap, bp, 
                                              tol=ls_tol, find_maxima=find_maxima,
                                              one_in_k=one_in_k, verbose=verbose)
            if verbose and find_maxima:
                if H_w != Ho_w:
                    print(':: Local maxima:', H_w, '\n')
            i += 1
            i0 += 1
            svns_cond = __svns_score(H_w, Ho_w,
                                     H | H_fs, Ho | Ho_fs,
                                     k) > 1 + theta if svns else False
            if H_w > Ho_w or svns_cond:
                delta_w = (H_w-Ho_w) / Ho_w * 100 
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                print(':: iteration: {}, distance in iterations to ' \
                      'earlier update: {}\n{}'.format(i, i0, 50*'--'))
                i0 = 0
                Ho_w = H_w
                Ho = H
                Ho_fs = H_fs
                ao = ap.copy()
                bo = bp.copy()
                k_cur = k_lims[0]
                hss.append(Ho)
                hfs.append(Ho_fs)
                run_trace.append((H_w, i))
            else:
                k_cur += k_step
            stop = (i >= max_iter or i0 >= max_iter_upd or process_time() - t0 >= timetol)
                
    delta_t = process_time()-t0
    print(':: Run completed @ {} ({} iterations), final f value: {:.6f} ({:.6f} per node)'
          .format(str(td(seconds=delta_t)), i, Ho_w, Ho_w / k))
    
    if svns:
        max_idx = np.argmax([hw for hw,i in run_trace])
        Ho_w = run_trace[max_idx][0]
        Ho = hss[max_idx]
        Ho_fs = hfs[max_idx]
    
    local_maximas_h = [np.where(h)[0] for h in hss]
    local_maximas_h_fs = [np.where(h)[0] for h in hfs]
    
    params = {'k':k,'A':A,'fss':fss,'k_lims':k_lims,'k_step':k_step,'timetol':timetol,
              'ls_tol':ls_tol,'ls_mode':ls_mode,'init_mode':init_mode,'beta_ratio':beta_ratio,
              'w_quantile':w_quantile,'seed':seed,'max_iter':max_iter,'max_iter_upd':max_iter_upd,
              'init_solution':init_solution,'theta':theta,'svns':svns,'one_in_k':one_in_k}
    
    run_vars = {'H':np.where(Ho)[0], 'H_fs':np.where(Ho_fs)[0], 'obj_score':Ho_w,
                'local_maximas_h':local_maximas_h, 'local_maximas_h_fs':local_maximas_h_fs, 
                'run_trace':run_trace, 'running_time':delta_t, 'iterations':i, 'params':params}
    
    return run_vars