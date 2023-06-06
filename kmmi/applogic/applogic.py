import numpy as np
import networkx as nx
from numba import *

from kmmi.constraints import *
from kmmi.enumeration import *
from kmmi.exposure import *
from ovns.ovns import *
from kmmi.pruning import *

import time
from datetime import timedelta

def kMMI(G, k, f=0.2, f_vs=None, Vs=None, n_sel=8000, n_min=1000, k_min=3, k_max=4,
         ptol=0.01, max_iter=2e5, max_iter_upd=1e5, nbtw=True, timetol=600,
         seed=4444):
    """
    Executes single run of kmmi algorithm.

    Parameters:
    G (NetworkX graph): Input graph.
    k (int): k parameter for the kMMI algorithm.
    f (float): Fraction of exposure m (default: 0.2).
    Vs (list, optional): List of nodes. If not provided, all nodes of the graph are used.
    n_sel (int, optional): Maximum number of subgraphs to select. Default is 8000.
    n_min (int, optional): Minimum number of subgraphs required. Default is 1000.
    k_min (int, optional): Lower bound for graphlet size. Default is 3.
    k_max (int, optional): Upper bound for graphlet size. Default is 4.
    ptol: tolerance parameter for the n_ranked selection
    max_iter (int): Maximum (total) number of iterations for the OVNS
                    algorithm (stopping criteria).
    max_iter_upd (int): Maximum number of iterations after last successful update
                        of the OVNS algorithm (stopping criteria).
    nbtw (bool): determines if non-backtracking mode for the Katz transformation
                 is used. (default: True)
    timetol (int, optional): Time tolerance. Default is 600.
    seed (int, optional): Seed for the random number generator. Default is 4444.

    Returns:
    tuple: Contains results of the OVNS algorithm.
    """

    print(50*'==')
    print(":: PHASE 1/5: INITIALIZE GRAPH AND PARAMETERS\n")

    if Vs is None:
        Vs = list(G.nodes)

    if n_min < k:
        n_min = 2*k

    rng = np.random.default_rng(seed=seed)
    N = len(G)
    A = nx.to_numpy_array(G)
    print(':: Initializing graph... Average degree: Vs{:.2f}, Number of seed nodes: {}'.format(2*len(G.edges) / len(G), len(Vs)))

    #### 2. Generate graphlet candidates

    print(50*'==')
    print(":: PHASE 2/5: GENERATE GRAPHLET CANDIDATES\n")

    t0 = time.time()
    U = generate_candidate_subgraphs_esu(G, Vs, k_min, k_max)
    t1 = time.time()
    delta_s = t1 - t0
    delta_td = timedelta(seconds=delta_s)
    U = to_numpy_array(U)
    print(':: Enumerating all connected graphlets...')
    print('* Number of graphlets generated:', U.shape[0])
    print(":: Elapsed time:", str(delta_td))
    node_counts(Vs, A, U)

    #### Handle case where not enough graphlets are generated
    if len(U) < n_min:
        for i in range(1,3):
            k_max += 1
            print(':: Not enough graphlets, k_max is too restricted, relax by 1')
            print(f'k_max set to {k_max}')

            U = generate_candidate_subgraphs_esu(G, Vs, k_min, k_max)
            U = to_numpy_array(U)
            if len(U) >= n_min:
                break

        node_counts(Vs, A, U)
        if len(U) < n_min:
            raise Exception('Not enough graphlets generated even after relaxing k_max')

    #### 3. Prune if necessary
    print(50*'==')
    print(":: PHASE 3/5: PRUNE GRAPHLETS IF NECESSARY\n")

    Uc = U
    if Uc.shape[0] >= n_sel:
        if Uc.shape[0] > n_sel:
            print(':: Pruning subgraphs...')
            Uc_sp = prune_subgraphs(Uc, k_min, k_max)
            node_counts(Vs, A, Uc_sp)
            Uc = Uc_sp if Uc_sp.shape[0] > n_min else Uc

        if Uc.shape[0] > n_sel:
            print(f':: Executing N_ranked selection, with n_sel={n_sel}')
            ws, ds = graphlet_scores(Uc, A)
            idxs = np.argsort(ws)[::-1]
            p_min, p_max = 5, 10
            Vs_uc = unique_nodes(Uc, A.shape[0])
            while True:
                Uc_nrank, _, C, _ = select_nrank(Uc[idxs,:], A, Vs_uc, p_min, p_max, ptol, n_iters=1,
                                           presorted=True, verbose=True, adaptive_p=True)
                if Uc_nrank.shape[0] >= n_min:
                    Uc = Uc_nrank
                    node_counts(Vs, A, Uc)
                    break
                p_min -= 1

    print(':: Selection ready')
    node_counts(Vs, A, Uc)
    Vs_uc = unique_nodes(Uc, A.shape[0])
    #### 4. Exposure matrix transformation

    print(50*'==')
    print(":: PHASE 4/5: EXPOSURE MATRIX TRANSFORMATION\n")

    E = to_exposure_matrix(G, f, verbose=True)
    E_hat, Ew, En_hat = coarse_grain_exposure_matrix_w(Uc, E, use_mutual_exposure=True)

    # ### 5. Transform to HkSP and solve
    print(50*'==')
    print(":: PHASE 5/5: TRANSFORM TO HkSP AND SOLVE\n")

    n = Uc.shape[0]
    k_lims = (1, np.min([k, n-k]))
    k_step = k // 10
    use_pref_attachment = True
    verbose = False

    E_input = transform_to_hks_input(E_hat)
    results = OVNS(k=k,
                   A=E_input,
                   k_lims=k_lims,
                   k_step=k_step,
                   timetol=timetol,
                   use_pref_attachment=use_pref_attachment,
                   max_iter=max_iter,
                   max_iter_upd=max_iter_upd,
                   verbose=verbose,
                   seed=seed
                   )

    print(':: kMMI Algorithm run complete.')
    return results, E, k, E_hat, Ew, En_hat, Uc, Vs_uc