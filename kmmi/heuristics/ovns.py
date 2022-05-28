from time import process_time
from datetime import timedelta as td
import numpy as np
import timeit
from numba import *
from kmmi.utils.utils import sub_sum
from kmmi.heuristics.initialize import *
from kmmi.heuristics.neighborhood_change import *
from kmmi.heuristics.neighborhood_search import *
from kmmi.heuristics.utils import __create_beam_array, __svns_score

def __auto_parametrize(H, H_w, A, A_beam, A_as, mu_beam_width, k, k_lims, k_step, 
                       ap, bp, ls_tol, find_maxima, one_in_k, w_quantile, 
                       v_target=1e4/60, verbose=False):
    """Logic for adjusting params to ensure effective iteration speeds. Used when
    'auto_parametrize' is set True for OVNS. 
    """
    n = A.shape[0]
    wq_min = np.max([1e-3, 10/n])
    determine_v = lambda: 10*timeit.timeit(lambda: ls_one_n_beam(H, H_w, A, A_beam, 
                                           alpha=ap, beta=bp, tol=ls_tol, 
                                           find_maxima=find_maxima, 
                                           one_in_k=one_in_k), 
                                           number=10)**-1
    if k_lims[1] > n-k:
        if verbose: print(':: WARNING: upper limit {} of the k_lims is above the ' \
                          ' range of allowed perturbation size values.'.format(k_lims[1]))
        k_lims = (k_lims[0], np.min([k, n-k]))
        print('::      --> readjusted to {}.'.format(k_lims))
    
    if n*k > 1e5:
        k_step = np.max([k_lims[1]//10, 5])
        if verbose:
            print(':: k_step set to {}'.format(k_step))

    v = determine_v()
    if verbose: print(':: Initial velocity: {:.4f} iter/s'.format(v))
    if v < v_target and find_maxima:
        find_maxima = False
        v = determine_v()
        if verbose:
            print(':: \'First\' local search mode enabled @ ({:.4f} iter/s)'.format(v))

    if v < v_target and not one_in_k:
        one_in_k = True
        v = determine_v()
        if verbose:
            print(':: \'One in k\' mode enabled @ ({:.4f} iter/s)'.format(v))

    while w_quantile / 2 > wq_min:
        v = determine_v()
        if v >= v_target: break
        w_quantile = w_quantile / 2
        w_thres = np.quantile(A[A != 0.0], 1.0-w_quantile)
        A_beam, mu_beam_width = __create_beam_array(A, A_as, w_thres)
    
    if verbose: 
        print(':: \'w_quantile decreased to {}\' @ ({:.4f} iter/s)'.format(
                w_quantile, v))
        print(':: Mean beam width: {:.2f}'.format(mu_beam_width))
        
    return find_maxima, one_in_k, k_step, w_quantile, A_beam, mu_beam_width, k_lims

def OVNS(k: int, A: np.array, k_lims: tuple, k_step: int=1, timetol: int=300, ls_tol: float=0.0, 
         ls_mode='best', use_pref_attachment=True, init_mode='drop-initial', beta_ratio: float=0.75, 
         seed: int=None, max_iter: int=2000000, max_iter_upd: int=1000000, w_quantile=1.0, 
         init_solution: np.array=None, svns=False, theta: float=0.06, one_in_k=False, 
         auto_parametrize=True, verbose=False): 
    """Opportunistic Variable Neighborhood Search heuristic for the HkSP. 
    
    Parameters
    ----------
    k : int
        Value of k in HkS (number of treatment groups).
    A : numpy.array (symmetric)
        Weighted adjacency matrix of the input network.
    k_lims : tuple of int
        Range for search depth, in general bounds are (0,k) or (0,n-k) if k > 0.5*dim(A).
        See also 'auto_parametrize'.
    k_step : int 
        Defines how many search depth steps to increment at once after unsuccessful update
        (default 1). See also 'auto_parametrize'.
    timetol : int
        Set the time upper bound (in seconds) for single run, (default 300)
    ls_tol : float, optional
        Set a tolerance for update (update is approved if improvement over current best 
        solution is at least ls_tol).
    ls_mode : string, optional
        Set local search mode. (default 'best')
        
        - 'best' : best improvement mode does exhaustive search over the 
                   space of all N1 neighbors and returns the best improvement.
        - 'first' : first improvement begins like 'best' but returns on first 
                    found improvement which allows faster exploration. Useful 
                    for large networks and high k values.
    one_in_k : bool, optional
         If True, selects one of the k nodes in the existing solution uniformly in random
         for replacement. Enabling allows more rapid exploration. See also 'auto_parametrize',
         (default False).
    use_pref_attachment : bool, optional
         When True perturbation will be biased towards high degree nodes similarly 
         as in preferential attachment in BA random network model (default True).
    init_mode : string, optional
         Select initialization strategy ('drop-initial','heaviest-edge','random','weighted-deg')
         (default 'drop-initial')
         
         - 'drop-initial' : start with all n nodes in the solution and iteratively
                            remove the node that contributes least to the solution.  
         - 'heaviest-edge' : selects k-x heaviest edges such that number of 
                             nodes in the solution amount to k.
         - 'random' : fully random initialization, selects k nodes uniformly in random.
         - 'weighted-deg' : selects k nodes based on the linear combination 
                            of their degree and weighted degree ranking. 
                            'beta_ratio' adjusts the weighting.
        See also 'init_solution'.
    beta_ratio : float, optional
         Sets the weights for the `weighted-deg` initialization strategy in range
         [0,1.0] where 0 gives all weight to degree ranking and 1 to weighted 
         degree ranking, (default 0.75).
    init_solution : numpy.array or list, optional
         Initial solution as a k length sequence of node ids (int) corresponding to A indeces 
         (default None).
    seed : int, optional
         Seed for the random number generator (note: due to lack of numba support for numpy, 
         some numpy.random library calls don't allow seed input currently). (default None)
    w_quantile : float, optional
         Define the quantile of heaviest edge weights that will be explored during each 
         neighorhood search (also "beam width"), lower values increase exploration speed.
         (default 1.0). See also 'auto_parametrize'.
    svns : bool, optional
         Enable svns mode. See also 'theta' for adjusting the search sensitivity.
    theta : float, optional
         Search sensitivity for svns mode. Conditional on 'svns' parameter being True.
    max_iter : int, optional
         Set the stopping criteria as number of max iterations (also optimization cycles)
         (default 2e6).
    max_iter_upd : int, optional
         Set the convergence criteria as maximum number of iterations (also, optimization 
         cycles) after last successful update (default 1e6).
    auto_parametrize : bool, optional
         If True, following parameters will be adjusted using simple heuristics to achieve minimum 
         speed of 1e4 optimization cycles per minute: (ls_mode, one_in_k, w_quantile, k_step), 
         (default True).
         
         Note! the auto_parametrize selection heuristic is based on early hyperparameter 
         optimization runs. However, they might not generalise to your use case. For best 
         results, it is advisable to run proper hyperparamater optimization for your 
         particular data set.
    verbose : bool
         Enables verbose mode (default False).
    
    Returns
    -------
    run_vars : dict 
         Dict with following keys:
         
         - H : numpy.array
             Best found approximate solution for the HkSP as a as k length sequence of node ids 
             (int) corresponding to indeces in the A matrix.
         - obj_score : float 
             Objective function value for best solution, f(H).
         - local_maximas_h : numpy.array with shape (x,n)
             History of all x solution updates for the executed run as boolean vectors of length n.
         - run_trace : list of tuples with length x
             List with objective score function values (tuple index 0) and iteration index 
             (tuple index 1) for the executed run.
         - running_time : int
             Running time in seconds, measured as process time (not wall time).
         - iterations : int
             Number of iterations (also optimization cycles).
         - params : dict
             Dictionary that includes all run parameters.
         - converged : bool
             True if run satisfied the convergence criteria set by 'max_uter_upd'.
         
    Examples
    --------
    Simple example with a random 1000x1000 adjacency matrix.
    
    >>> n = 1000
    >>> A = np.random.random((n,n))
    >>> # Remove self-edges
    >>> A[np.diag_indices_from(A)] = 0.0
    
    >>> k = 16
    >>> k_lims = (1,k)
    >>> timetol = 10

    >>> run = OVNS(k, A, k_lims, timetol=timetol)
    :: Automatic parametrization enabled.
    :: Initialization and first local search completed.
    :: LS took 0:00:00.178016, value: 78.0250735429751

    :: Found new maxima: 80.448778, change: +3.11%
    :: iteration: 1, distance in iterations to earlier update: 1
    -----------------------------------------------------------------    
    :: Found new maxima ... ... ...
    -----------------------------------------------------------------
    :: Found new maxima: 159.379325, change: +0.88%
    :: iteration: 3790, distance in iterations to earlier update: 545
    -----------------------------------------------------------------
    :: Run completed @ 0:00:10.001267 (6430 iterations), 
    :: final f value: 159.379325 (9.961208 per node)
    
    >>> print('Selected params: ', run['params'])
    Selected params:  {'k': 16, 'k_lims': (1, 16), 'k_step': 1, 
                       'timetol': 10, 'ls_tol': 0.0, 
                       'ls_mode': 'best', 'use_pref_attachment': True, 
                       'init_mode': 'drop-initial', 'beta_ratio': 0.75, 
                       'seed': None, 'max_iter': 2000000, 
                       'max_iter_upd': 1000000, 'w_quantile': 1.0, 
                       'init_solution': None, 
                       'theta': 0.06, 'svns': False, 
                       'one_in_k': False, 'auto_parametrize': True}
    
    >>> print('Best solution: ', run['H'])
    Best solution:  [ 22  74 126 192 238 364 421 518 527 599 667 673 723 781 866 884]
    
    >>> print('Best solution f value: ', run['obj_score'])
    Best solution f value:  159.37932529642183
    """
    t0 = process_time()
    n = A.shape[0]
    assert ls_mode in ['first','best'], "Invalid local search mode; choose 'first' or 'best'" 
    assert init_mode in ['heaviest-edge', 'weighted-deg', 'random', 'drop-initial']
    assert 0.0 < w_quantile <= 1.0, 'Invalid value, select value from range (0.0,1.0]'
    assert k != n, 'Input k value equals n, solution contains all nodes in the network'
    assert k < n, 'Input k value is greater than n; select k such that k < n'
    
    find_maxima = ls_mode == 'best'
    w_thres = np.quantile(A[A != 0.0], 1.0-w_quantile)
    A_as = np.argsort(A)[:,::-1]
    A_beam, mu_beam_width = __create_beam_array(A, A_as, w_thres)
    Ho, Ho_w, ao, bo, p_w = initialize_solution(A, A_beam, k, init_solution, init_mode, ls_tol, 
                                                beta_ratio, find_maxima, one_in_k, 
                                                verbose=verbose)
    hss = [(Ho)]
    run_trace = [(Ho_w, 0)]
    if auto_parametrize:
        print(':: Automatic parametrization enabled.')
        find_maxima, one_in_k, k_step, w_quantile, A_beam, mu_beam_width, k_lims = \
        __auto_parametrize(
            Ho, Ho_w, A, A_beam, A_as, mu_beam_width, k, k_lims, k_step, ao, bo, 
            ls_tol, find_maxima, one_in_k, w_quantile, verbose=verbose
        )
        
        ls_mode = 'best' if find_maxima else 'first'
    
    assert mu_beam_width > 0, 'Set w_thres={:.4f}, A_beam shape {}, (no elements)' \
                                .format(w_thres, mu_beam_width)
    if mu_beam_width < 10:
        print(':: WARNING: determined beam width is narrow @ {}, optimization result ' \
              'might suffer, for better result try increasing w_quantile'
              .format(mu_beam_width))
    
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
                    print(':: Local maxima:', H_w, '\n')
            i += 1
            i0 += 1
            svns_cond = __svns_score(H_w, Ho_w, H, Ho, k) > 1 + theta if svns else False 
            if H_w > Ho_w or svns_cond:
                # 3. Update vars
                delta_w = (H_w-Ho_w) / Ho_w * 100
                print(':: Found new maxima: {:.6f}, change: +{:.2f}%'.format(H_w, delta_w))
                print(':: iteration: {}, distance in iterations to ' \
                      'earlier update: {}\n{}'.format(i, i0, 50*'--'))
                i0 = 0
                Ho_w = H_w
                Ho = H.copy()
                ao = ap.copy()
                bo = bp.copy()
                hss.append(Ho)
                run_trace.append((H_w, i))
                k_cur = k_lims[0]
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
    
    local_maximas_h = [np.where(h)[0] for h in hss]
    
    params = {'k':k,'k_lims':k_lims,'k_step':k_step,'timetol':timetol,'ls_tol':ls_tol,
              'ls_mode':ls_mode,'use_pref_attachment':use_pref_attachment,'init_mode':init_mode,
              'beta_ratio':beta_ratio,'seed':seed,'max_iter':max_iter,'max_iter_upd':max_iter_upd,
              'w_quantile':w_quantile,'init_solution':init_solution,'theta':theta,'svns':svns,
              'one_in_k':one_in_k,'auto_parametrize':auto_parametrize}
    
    converged = True if i0 >= max_iter_upd else False
    run_vars = {'H':np.where(Ho)[0], 'obj_score':Ho_w, 'local_maximas_h':local_maximas_h,
                'run_trace':run_trace, 'running_time':delta_t, 'iterations':i, 'params':params,
                'converged': i0 >= max_iter_upd}
    
    return run_vars