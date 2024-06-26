"""
@name: test_graphs.py                         
@description:                  
    Unit test module toolbox.graphs

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import pytest
import networkx as nx

import toolbox.graphs as graph 

G1 = nx.Graph()
for i in range(5): G1.add_edge(i,i+1,weight=1)
G2 = nx.Graph()
for i in range(2,10): G2.add_edge(i,i+1,weight=1)
H = [G1.copy(),G2.copy()]

def test_composite_nodes():
    exp = list(range(11))
    assert exp == graph.composite_nodes(H)

def test_consensus_graph():
    ## Degree 2 test
    M = graph.consensus(H,2)
    exp = [(2,3),(3,4),(4,5)]
    assert exp == list(M.edges())
    for (u,v,w) in M.edges(data='weight'): assert w == 1
    
    ## Degree 1 test
    M = graph.consensus(H,1)
    exp = [(0,1),(1,2),(5,6),(6,7),(7,8),(8,9),(9,10)]
    assert exp == list(M.edges())
    for (u,v,w) in M.edges(data='weight'): assert w == 1

def test_index_merge():
    M1 = graph.consensus(H,1)
    M2 = graph.consensus(H,2)

    M = graph.index_merge([M1,M2])
    
    id1 = [(0,1),(1,2),(5,6),(6,7),(7,8),(8,9),(9,10)]
    id2 = [(2,3),(3,4),(4,5)]

    for (u,v) in id1: assert 1 == M[u][v]['id']
    for (u,v) in id2: assert 2 == M[u][v]['id']

def test_zip_index_consensus_graph_edges():
    M1 = graph.consensus(H,1)
    M2 = graph.consensus(H,2)

    M = graph.index_merge([M1,M2])
    graph.zip_index_consensus_graph_edges(M,H)
    
    for (u,v) in M.edges():
        g_index = list(map(int,M[u][v]['g_index'].split('-')))
        g_index_weight = list(map(int,M[u][v]['g_index_weight'].split('-')))
        for (idx,w) in zip(g_index,g_index_weight):
            assert H[idx].has_edge(u,v)
            assert H[idx][u][v]['weight'] == w

def test_unzip_index_consensus_graph_edges():
    M1 = graph.consensus(H,1)
    M2 = graph.consensus(H,2)

    M = graph.index_merge([M1,M2])
    graph.zip_index_consensus_graph_edges(M,H)
    Hexp = graph.unzip_index_consensus_graph_edges(M)
    
    assert nx.utils.misc.graphs_equal(G1,Hexp[0])
    assert nx.utils.misc.graphs_equal(G2,Hexp[1])


if __name__=="__main__":
    test_unzip_index_consensus_graph_edges()
