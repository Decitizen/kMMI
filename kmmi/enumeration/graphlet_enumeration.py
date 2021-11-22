import pymnet as pn

def generate_candidate_subgraphs_esu(G, Vs, k_min, k_max, custom_processing_function=None):
    u"""Find all graphlets (connected induced subgraphs) with number of nodes k_min <= n <= k_max.
    NB! Treats the network as an undirected one, i.e. for directed graphs we use the 'weakly connected' notion of connectedness.
    if custom_processing_function is None:
        Returns: list, where each element is a list of nodes that form a graphlet
    else:
        calls custom_processing_function whenever a subgraph is found, with parameter ([nodelist],[layerlist])
        (layerlist not relevant here, nodelist contains the subgraph nodes)
        example: custom_processing_function = lambda sub: print(sub[0])
    """
    magenta_G = G.subgraph(Vs)
    M = pn.MultilayerNetwork(aspects=1,directed=False,fullyInterconnected=True)
    M.add_layer(0)
    for ii in magenta_G.nodes:
        M.add_node(ii)
    for u,v in magenta_G.edges:
        M[u,v,0] = 1
    if not custom_processing_function:
        reslist = []
        res = lambda sub : reslist.append(sub[0])
    else:
        res = custom_processing_function
    for k in range(k_min,k_max+1):
        pn.sampling.esu.sample_multilayer_subgraphs_esu(M,res,nnodes=k,nlayers=1)
    if not custom_processing_function:
        return reslist
