"""
@name: modify.py
@description:
    Functions for modifying graphs

Module for formating and manipulating graphs

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2020-03
"""

import networkx as nx
import numpy as np

def low_pass_edge_filter(G,pct,attr='weight'):
    """
    Remove edges with attr above therehold
    
    Inputs:
    -------
    G : networkx, directed or undirected
    pct: percentile threshold, float, between 0 and 1
    attr: str, edge attribute (deftaul 'weight')
    """
    weights = [w for (a,b,w) in G.edges.data(attr)]
    thresh = np.percentile(weights,pct)
    G.remove_edges_from([(a,b) for (a,b,w) in G.edges.data(attr) if w >= thresh])
    G.remove_nodes_from(list(nx.isolates(G)))

def band_pass_edge_filter(G,pct,attr='weight'):
    """
    Remove edges with attr below low and above high threshold
    
    Inputs:
    -------
    G : networkx, directed or undirected
    pct: percentile threshold, list, [low threshold, high threshold]
    attr: str, edge attribute (deftaul 'weight')
    """
    weights = [w for (a,b,w) in G.edges.data(attr)]
    thresh1 = np.percentile(weights,pct[0])
    thresh2 = np.percentile(weights,pct[1])
    edges = [(a,b) for (a,b,w) in G.edges.data(attr) if (w < thresh1 or w >= thresh2) ]
    G.remove_edges_from(edges)
    G.remove_nodes_from(list(nx.isolates(G)))

def high_pass_edge_filter(G,pct,attr='weight'):
    """
    Remove edges with attr above therehold
    
    Inputs:
    -------
    G : networkx, directed or undirected
    pct: percentile threshold, float, between 0 and 1
    attr: str, edge attribute (deftaul 'weight')
    """
    weights = [w for (a,b,w) in G.edges.data(attr)]
    thresh = np.percentile(weights,pct)
    G.remove_edges_from([(a,b) for (a,b,w) in G.edges.data(attr) if w < thresh])
    G.remove_nodes_from(list(nx.isolates(G)))

def filter_graph_edge(A,pct=50,weight='weight',thresh_high=True):
    """
    Remove edges from graphs that do satify edge threshold

    Input:
    ------
    A : input networkx graph
    pct: int, edge percentile threshold
    thresh_high: bool (default=True), if true pct is the upper threshold,
        if false, pct is the lower threshold

    Return:
    -------
    H : modified graph
    """
    H = nx.Graph()
    nodes = sorted(A.nodes())
    weights = [w for (u,v,w) in A.edges(data=weight)]
    thresh = np.percentile(weights,pct)
    for (u,v) in A.edges():
        if thresh_high: 
            c = thresh > A[u][v][weight]
        else:
            c = thresh <= A[u][v][weight]
        if c: continue
        H.add_edge(u,v)
        for (a,b) in A[u][v].items(): H[u][v][a] = b
    return H

def normalize_edge_weight(A,weight='weight'):
    """
    Normalize edge weights by the total weight in the graph
    
    Adds edge attribute 'wnorm' to the graph

    Input:
    ------
    A : input networkx graph
    
    """
    tot = np.sum([w for (u,v,w) in A.edges(data=weight)])
    for (u,v) in A.edges(): A[u][v]['wnorm'] = A[u][v][weight] / tot
     
 
def standardize_edge_weigth(G,attr='weight'):
    """
    Standardizes the edge weight of teh graph

    Log-normalizes the edge weights and stardardizes to set mean to x = 0

    Input:
    -----
    G: input networkx graph

    """
    weight = [np.log(w) for (a,b,w) in G.edges(data=attr)]
    mu = np.mean(weight)
    std = np.std(weight)
    for (a,b,w) in G.edges(data=attr):
        G[a][b][attr] = (np.log(w) - mu) / std
 
def graph_to_array(G,attr=['weight']):
    """
    Convert graph to edge array

    Input:
    ------
    G: input graphs
    attr: list of attributes

    Return:
    -------
    arr : [n,3] numpy array where n is the number of edges, row format
    
    """
    arr = np.zeros((G.number_of_edges(),len(attr)))
    for (i,(a,b)) in enumerate(G.edges()):
        for (j,atr) in enumerate(attr): 
            arr[i,j] = G[a][b][atr]
    return arr

def clean_graph(G,Ref):
    """
    Returns graph that has edges in both G and Ref

    Inputs:
    -------
    G : networkx, directed graph
    Ref: networkx, graph
    
    Return:
    -------
    H : networkx graph, same type as G with node and edge attributes 
    """
    H = G.copy()
    H.remove_nodes_from([n for n in G if n not in Ref])
    H.remove_edges_from([(a,b) for (a,b) in G.edges() if not Ref.has_edge(a,b)])
    return H


def clean_ref_graph(Ref,A):
    """
    Removed edges in Ref graph not in A

    Parameters:
    -----------
    Ref: networkx Graph, reference graph
    A: networkx Graph, graph for screening edges

    Return:
    -------
    Cleaned copy of Ref graph
    """
    H = nx.Graph()
    if Ref.is_directed(): H = nx.DiGraph()
    for (a,b) in Ref.edges():
        if A.has_edge(a,b):
            H.add_edge(a,b,weight=Ref[a][b]['weight'],id=Ref[a][b]['id'])
    return H

def subgraph(G,nodes):
    """
    Returns subgraph only with nodes
    """
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in nodes)
    SG.add_edges_from((n, nbr, d)
            for n, nbrs in G.adj.items() if n in nodes
            for nbr, d in nbrs.items() if nbr in nodes)
    SG.graph.update(G.graph)
    return SG

def split_graph(G,nodes):
    """
    Return subgraph with edges induced by nodes
    """
    H = G.__class__() 
    for n in nodes:
        if not G.has_node(n): continue
        for m in G.neighbors(n):
            H.add_edge(n,m,weight=G[n][m]['weight'])
        if G.is_directed():
            for m in G.predecessors(n):
                H.add_edge(m,n,weight=G[m][n]['weight'])
    return H


def map_graph_nodes(G,nmap):
    """
    Maps left and right nodes of graph

    G : networkx graph
        Graph
    nmap : dict
        Dictionary that maps node names
    """
    H = G.__class__()
    H.add_edges_from((nmap[n], nmap[nbr], d) 
            for n, nbrs in G.adj.items() 
            for nbr, d in nbrs.items())
    return H
    
def apply_mask(Mask,G):
    rm_edges = [(a,b) for (a,b) in G.edges() if not Mask.has_edge(a,b)]
    G.remove_edges_from(rm_edges)
    return G

def collapse_nodes(G,nmap):
    H = G.__class__()
    for (u,v) in G.edges:
        for (k,w) in G[u][v].items():
            um = nmap[u]
            vm = nmap[v]
            if not H.has_edge(um,vm): H.add_edge(um,vm)
            try:
                H[um][vm][k] += w
            except:
                H[um][vm][k] = w
    return H

