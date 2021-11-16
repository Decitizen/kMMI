import pymnet as pn

def generate_candidate_subgraphs_esu(G, Vs, k_min, k_max):
    u"""Find all graphlets (connected induced subgraphs) with number of nodes k_min <= n <= k_max.
    NB! Treats the network as an undirected one, i.e. for directed graphs we use the 'weakly connected' notion of connectedness.
    Returns: list, where each element is a list of nodes that form a graphlet
    """
    magenta_G = G.subgraph(Vs)
    M = pn.MultilayerNetwork(aspects=1,directed=False,fullyInterconnected=True)
    M.add_layer(0)
    for ii in magenta_G.nodes:
        M.add_node(ii)
    for u,v in magenta_G.edges:
        M[u,v,0] = 1
    res = []
    for k in range(k_min,k_max+1):
        pn.sampling.esu.sample_multilayer_subgraphs_esu(M,res,nnodes=k,nlayers=1)
    return [r[0] for r in res]