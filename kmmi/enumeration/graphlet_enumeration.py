import igraph as ig
import networkx as nx

def generate_candidate_subgraphs_esu(G, Vs, k_min, k_max, w_thres=0.0):
    """Find all graphlets (connected induced subgraphs) with number of nodes 
    k_min <= n <= k_max. NB! Treats the network as an undirected one, i.e. 
    for directed graphs we use the 'weakly connected' notion of connectedness.
    """
    Gt = nx.Graph()
    for u,v,w in G.edges(data='weight'):
        if (v,u) in Gt.edges: continue
        if w >= w_thres and u in Vs and v in Vs:
            Gt.add_edge(u,v)
    
    n = len(Gt)
    node_map = dict(zip(Gt.nodes, range(n)))
    rnode_map = dict(zip(range(n), Gt.nodes))
    Gs = nx.relabel_nodes(Gt, node_map)
    G_ig = ig.Graph.from_networkx(Gs)
    
    all_motifs = []
    for k in range(k_min,k_max+1):
        motifs = []
        G_ig.motifs_randesu(size=k, callback=lambda g, ids, iclass: motifs.append(ids))
        all_motifs += motifs
    return [[rnode_map[u] for u in motif] for motif in all_motifs]