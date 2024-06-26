"""
@name: test_graphs_modify.py                       
@description:                  
    Unit testing for toolbox.graphs.modify

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import networkx as nx

import toolbox.graphs.modify as gm

def build_graph():
    G = nx.Graph()
    for i in range(1,101): G.add_edge(f'u_{i}',f'v_{i}',weight=i)
    return G

def test_low_pass_edge_filter():
    G = build_graph() 
    gm.low_pass_edge_filter(G,35)
    assert G.number_of_edges() == 35
    for (u,v,w) in G.edges(data='weight'): assert w <= 35
    for n in G.nodes(): assert G.degree(n) > 0

def test_high_pass_edge_filter():
    G = build_graph() 
    gm.high_pass_edge_filter(G,35)
    assert G.number_of_edges() == 65
    for (u,v,w) in G.edges(data='weight'): assert w > 35
    for n in G.nodes(): assert G.degree(n) > 0

def test_band_pass_edge_filter():
    G = build_graph() 
    gm.band_pass_edge_filter(G,[25,75])
    assert G.number_of_edges() == 50
    for (u,v,w) in G.edges(data='weight'): 
        assert w > 25
        assert w <= 75
    for n in G.nodes(): assert G.degree(n) > 0

def test_normalize_edge_weight():
    G = nx.Graph()
    for i in range(10): G.add_edge(f'u_{i}',f'v_{i}',weight=1)
    gm.normalize_edge_weight(G)
    for (u,v,w) in G.edges(data='wnorm'): assert w == 0.1
