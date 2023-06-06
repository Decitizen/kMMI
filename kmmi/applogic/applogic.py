import numpy as np
import networkx as nx
from numba import *

from kmmi.constraints import *
from kmmi.enumeration import *
from kmmi.exposure import *
from ovns.ovns import *
from kmmi.pruning import *

#FIXME: update to use the refactored code
def run(G: nx.DiGraph, 
        Vs: list, 
        pos: dict, 
        k: int, 
        f: float=0.85, 
        k_min: int=3, 
        k_max: int=4,
        delta_min: float=0.5, 
        delta_max: float=1.0, 
        w_threshold: float=1e-4, 
        expected: list=[],
        force_select: list=[],
        penalty_weight: float=4.0,
        weight_attr: str='weight',
        timetol: int=600,
        beta_ratio: float=0.25,
        ls_mode: str='first',
        w_quantile: float=0.99,
        seed: int= 2222,
        pre_enumerate_timetol: int=20,
        nbtw=False,
        remove_small_seeds=False,
        visualize=False,
        verbose=False
        ):
    
    """Executes single run of kmmi algorithm.
    
    Returns:
    GVs (nx.Graph): final graph
    E_directed (np.array): coarse grained exposure (adjacency) matrix  
    [fs, ds, es]: solution overlaps
    """ 
    assert len(set(force_select)) == len(force_select), 'Remove duplicates from the force selection nodes.'
    
    if verbose: 
        print(50*'==')
        print("KMMI INITIALIZED\n")
        if nbtw: print(':: Non-backtracking running mode enabled.')
        if len(force_select) > 0: print(':: Forced selection running mode enabled.')
        print(':: Pre-process by max-normalizing input network weights')
    
    if weight_attr != 'weight':
        values = dict(zip(G.edges, [G.edges[e][weight_attr] for e in G.edges]))
        nx.set_edge_attributes(G, values, 'weight')
    
    # Max-normalize weights in the underlying graph
    w_max = max(G.edges(data='weight'), key=lambda x: x[2])
    for e in G.edges:
        G.edges[e]['weight'] /= w_max[2]
        
    ##  1. Graphlet generation
    if verbose: 
        print(50*'==')
        print("PHASE 1/5: GENERATE GRAPHLETS\n")
        print(':: Enumerating all connected graphlets')
        
    #### 1.1. Generate graphlets       
    Vs_hat = generate_candidate_subgraphs_esu(G, Vs, k_min=k_min, k_max=k_max)
    if verbose: print(':: Total number of graphlets generated: {}'.format(len(Vs_hat)))   
    Vs_hat_augmented = prune(G, Vs_hat=Vs_hat, 
                             delta_min=delta_min, 
                             delta_max=delta_max,
                             k_min=k_min, k_max=k_max,
                             force_select=force_select, 
                             remove_small_seeds=remove_small_seeds, verbose=verbose)

    n_eff = len(set([v for s in Vs_hat_augmented for v in s]))
    print(':: Number of unique g-nodes after pruning: ', n_eff)
                    
    ##  2. Compute exposure matrix
    if verbose: print(50*'==')
    if verbose: print("PHASE 2/5: COMPUTE EXPOSURE MATRIX\n")
    E_directed = to_exposure_matrix(G, f, w_threshold = w_threshold, 
                                    nbtw=True, verbose=verbose)
       
    ##  3. Coarse grain
    if verbose: 
        print(50*'==')
        print("PHASE 3/5: COARSE GRAIN EXPOSURE MATRIX\n")
        print(':: Coarse graining exposure matrix (penalty weight: {})'.format(penalty_weight))

    U = to_numpy_array(Vs_hat_augmented)
    E_hat = coarse_grain_exposure_matrix(U, E_directed, 
                                         use_mutual_exposure=True, 
                                         penalty_weight=penalty_weight)

    ##  4. Find near-optimal solution to HkS
    if verbose: 
        print(50*'==')
        print("PHASE 4/5: TRANSFORM INPUT AND SOLVE WITH OVNS\n")
        print(':: Transforming to HkS input format (edgelist)')
    
    ## 4.1 Transform to HkS input
    GVs = transform_to_hks_input(E_hat)
    if visualize:
        ws = compute_ghat_penalty_weight_distribution(GVs, Vs_hat_augmented)
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.hist(ws)
        ax.set_xlim(0,1.0)
        ax.set_title('Penalty weight distribution after coarse graining and input transformation')
        
    if not nx.is_connected(GVs.to_undirected()): print('WARNING: network is not connected.')
    
    if visualize:
        fig, axes = plt.subplots(1,3,figsize=(24,8))
        axes = axes.ravel()
        if force_select:
            visualize_ghat_dummy_nodes(GVs, node_map, ax=axes[0], u_prime_map=u_prime_map)
    
    if verbose: 
        print(':: Computing heaviest {}-subgraph with {} pre-selected nodes\n'.format(k, 
                                                                                      len(force_select)))
    
    ## 4.1 Pre-enumerate force selected node configurations
    n = len(GVs)
    Ec, ol_fs_ids, fs_ids, fs_ids_rev = to_dependency_network(Vs_hat_augmented, force_select)
    Gec = nx.Graph(Ec)
    Gec_sub = nx.subgraph(Gec, fs_ids.keys())
    ss = sample_independent_sets(Gec_sub, force_select, fs_ids, fs_ids_rev, 
                                 target_time=pre_enumerate_timetol, 
                                 verbose=verbose)
    
    ##  4.2 Run OVNS
    ss = [list(s) for s in ss]
    k_lims = (1, np.min([k, n-k]))
    k_step = 1
    A = nx.to_numpy_array(GVs)
    H_opt, H_opt_fs, H_opt_w = OVNSfs(k, A, ss, 
                                      k_lims, k_step, 
                                      timetol, 0.0, 
                                      ls_mode, beta_ratio, 
                                      seed, w_quantile, 
                                      verbose=False)
    
    if verbose: print(':: k={}, W^(H)={:.3f}, H_opt={}, H_opt_fs={}'.format(k, 
                                                                             H_opt_w, 
                                                                             H_opt, 
                                                                             H_opt_fs))
    
    H_opt_mapped = [Vs_hat_augmented[s] for s in H_opt]
    H_opt_fs_mapped = [Vs_hat_augmented[s] for s in H_opt_fs]
    
    return H_opt_mapped, H_opt_fs_mapped, H_opt_w

