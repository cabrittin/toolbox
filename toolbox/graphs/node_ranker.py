"""
@name: node_ranker
@description:
    Given graph ranks nodes

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""
import networkx as nx
import numpy as np


def get_graph_rank(G,ranker,**kwargs):
    if isinstance(G,str): G = nx.read_graphml(G)
    return ranker(G,**kwargs)

def get_standard_rank(gpath,ranker,log_rank=False,**kwargs):
    rank = get_graph_rank(gpath,ranker,**kwargs)
    if log_rank: rank = log_transform_rank(rank)
    return standardize_rank(rank)

def get_rank(gpath,ranker,log_rank=False,**kwargs):
    rank = get_graph_rank(gpath,ranker,**kwargs)
    if log_rank: rank = log_transform_rank(rank)
    return rank
 
def degree(G,**kwargs):
    rnk = sorted([(n,G.degree(n)) for n in G.nodes()],key=lambda x: x[1])
    return rnk

def degree_weight(G,**kwargs):
    rnk = sorted([(n,G.degree(n,weight='weight')) for n in G.nodes()],key=lambda x: x[1])
    return rnk

def eig_centrality(G,**kwargs):
    _rnk = nx.eigenvector_centrality(G,weight='weight')
    rnk = [item for item in sorted(_rnk.items(),key=lambda item: item[1])]
    return rnk

def second_order_centrality(G,**kwargs):
    _rnk = nx.second_order_centrality(G)
    rnk = [item for item in sorted(_rnk.items(),key=lambda item: item[1])]
    return rnk

def betweeness_centrality(G,**kwargs):
    H = nx.Graph() 
    for (u,v,w) in G.edges(data='weight'):
        H.add_edge(u,v,weight=1./w)
    _rnk = nx.betweenness_centrality(H,weight='weight')
    #_rnk = nx.betweenness_centrality(G)
    rnk = [item for item in sorted(_rnk.items(),key=lambda item: item[1])]
    return rnk

def fraction_degree_with_target(G,target_nodes=None):
    w = np.sum([w for (u,v,w) in G.edges(data='weight')])
    deg = dict([(n,G.degree(n,weight='weight')) for n in G.nodes()])
    _rnk = {}
    for n in G.nodes():
        _rnk[n] = 0
        for m in target_nodes:
            if not G.has_edge(n,m): continue
            _rnk[n] += G[n][m]['weight']
        #_rnk[n] = _rnk[n] / deg[n]
        _rnk[n] = _rnk[n] / w
        #if _rnk[n] == 0: _rnk[n] = 1e-6
        #_rnk[n] = np.sqrt(_rnk[n])
    rnk = [item for item in sorted(_rnk.items(),key=lambda item: item[1])]
    zero_rnk = [r for r in rnk if r[1] == 0]
    #print(zero_rnk)
    rnk = [ r for r in rnk if r[1] > 0] 
    return rnk 

def fraction_of_targets(G,target_nodes=None):
    w = len(target_nodes) 
    _rnk = {}
    for n in G.nodes():
        _rnk[n] = 0
        for m in target_nodes:
            if not G.has_edge(n,m): continue
            _rnk[n] += 1
        _rnk[n] = _rnk[n] / w
    rnk = [item for item in sorted(_rnk.items(),key=lambda item: item[1])]
    return rnk

def standardize_rank(rank,log=False):
    w = np.array([r[1] for r in rank])
    mu = np.mean(w)
    std = np.std(w)
    _rank = [(u,(v-mu)/std) for (u,v) in rank]
    return _rank

def log_transform_rank(rank):
    _rank = [(u,np.log(v)) for (u,v) in rank]
    return _rank


def sorted_target_split_rank(rank,target_nodes):
    target = [[],[]]
    for r in rank: target[int(r[0] in target_nodes)].append(r)
    for i in range(2): target[i] = sorted(target[i],key=lambda x: x[1])
    return target


