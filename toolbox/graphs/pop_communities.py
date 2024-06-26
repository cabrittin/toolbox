"""
@name: pop_communiteis                         
@description:                  
    Module for computing population based communities

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import networkx as nx
import networkx.algorithms.community as nx_comm 
from scipy.cluster.hierarchy import linkage,fcluster,dendrogram
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl

import toolbox.graphs.modify as gm
from toolbox.graphs import consensus
from collections import defaultdict

def pop_comm_generate(PG,pop_size,tqdm_disable=True,tqdm_desc='Iter',**kwargs):
    """
    Computes communities across a population generator

    Args:
    -----
    PG: Population generator class
    pop_size: int
        Number of individuals to draw from a population
    **kwargs: Kwargs for pop_comm, see pop_comm function for details 
    
    Returns:
    --------
    A [m,n] numpy int array C, where m is the pop_size and n is the number of nodes, and
    C[i,j] is the community assignment of the jth node in the ith individual from the population. 
    """
    nodes = kwargs.pop('nodes',None)
    if nodes is None: 
        G = PG.pop_population()
        nodes = sorted(G[0].nodes())
    ndict = dict([(n,i) for (i,n) in enumerate(nodes)])

    C = np.zeros([pop_size,len(nodes)])
    for i in tqdm(range(pop_size),desc=tqdm_desc,disable=tqdm_disable):
        cls = pop_comm(PG,**kwargs) 
        for (node,comm) in cls.items(): C[i,ndict[node]] = comm
    return C       
   

def pop_comm(PG,edge_thresh=None,delta=None,nodes=None,mask=None,algorithm='louvain_communities',**kwargs):
    """
    Compute communties for individual graph from a population generator.

    Args:
    -----
    PG: Population generator class
    edge_thresh: int, optional (default: None)
        If not None, percentile below which edge will be filtered out
    delta: int, optional (default: None)
        If not None, Number of graphs an edge need to be conserved in. If None, 
        the default is delta the number of graphs in the PopulationGeneratior
    nodes: list, optional (default: None)
        List of nodes to compute communities. If None, the nodes of the first graph
        of the population generator will be used.
    mask: list, optional (default: None)
        List of networkx edges. If not None, communties will be computed using only the provided edges
    algorithm: str, optional (default: louvain_communities)
        Networx community algorithm to use. 

    **kwargs: Any keyword arguments for the networkx community algorithm. 
    
    Returns:
    Dictions with (key,value) pairs, key:node label, value: communitiy assignment (int label)

    """

    G = PG.pop_population()
    
    if edge_thresh is not None: G = [gm.filter_graph_edge(g,edge_thresh) for g in G]
    for g in G: gm.normalize_edge_weight(g)
    
    if delta is None: delta = len(G)
    if nodes is None: nodes = sorted(G[0].nodes())
    
    M = consensus(G,delta,nodes=nodes,weight=['weight','wnorm'])
    
    if mask is not None: M = gm.apply_mask(F,M)
    
    comms = getattr(nx_comm,algorithm)(M,weight='wnorm',**kwargs)
    
    d = {} 
    for (cls_id,comm) in enumerate(comms):
        for n in comm: d[n] = cls_id
    return d

def pop_comm_correlation(C):
    """
    Computes the frequency that pairs of nodes are placed in the same
    community across a population

    Args:
    -----
    C : numpy array
        A [m,n] numpy int array C, where m is the pop_size and n is the number of nodes, and
        C[i,j] is the community assignment of the jth node in the ith individual from the population. 
    
    Returns:
    --------
    A [n,n] numpy array Z, where n is the number of nodes, and
    Z[i,j] is the frequency that nodes i and j are clustered together across a population.

    """
    num_nodes = C.shape[1]
    z = np.zeros([num_nodes,num_nodes]) 
    for (i,j) in combinations(range(num_nodes),2):
        s = (C[:,i] == C[:,j]).sum()
        z[i,j] = s
        z[j,i] = s

    z /= C.shape[0]
    np.fill_diagonal(z,1)
    return z 

def pop_comm_linkage(Z,method='ward',optimal_ordering=True):
    """
    Computes the frequency that pairs of nodes are placed in the same
    community across a population

    Primarily a convencience function to avoid having to import scipy module
    in calling script

    Args:
    -----
    Z : numpy array
        A [n,n] numpy array Z, where n is the number of nodes, and
        Z[i,j] is the frequency that nodes i and j are clustered together across a population.
    method: str, optional (default 'ward')
        Scipy methods for computing linkage correlation.
    optimal_ordering: bool, optional (default True)
        Use Scipy optimal ordering

    Returns:
    --------
    Scipy hierarchy linkage
    """
    return linkage(Z, method=method,optimal_ordering = optimal_ordering)


def pop_assign_comm(C,max_d,**kwargs):
    """
    Assigns clusters to nodes based on dendrogram distance max_d

    Args:
    -----
    C : numpy array
        A [m,n] numpy int array C, where m is the pop_size and n is the number of nodes, and
        C[i,j] is the community assignment of the jth node in the ith individual from the population. 
    max_d : int
        Dendrogram distance threshold, below which, nodes will be placed in the same cluster
    **kwargs: Kwargs for pop_comm_linkage 

    method: str, optional (default 'ward')
        Scipy methods for computing linkage correlation.
    optimal_ordering: bool, optional (default True)
        Use Scipy optimal ordering

    Returns:
    --------
    List of lists of clusters. Each sublist is a clusters. Each element is a sublist is the node label. 
    """
    z = pop_comm_correlation(C) 
    y = pop_comm_linkage(z,**kwargs)
    clusters = fcluster(y,max_d,criterion='distance')
    return clusters

def pop_comm_dendrogram(*args, **kwargs):
    """
    Plots the dendrogram used for clustering
    
    Taken from: 
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

    First arg should be the linkage. See pop_comm_corre

    """

    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def pop_comm_heatmap(z,y,yticklabels=[],colors=None,no_cbar=True,fig_title=None):
    """
    Heatmap of pop_comm_correlation and pop_comm_linkage
    """
    mpl.rcParams['ytick.labelsize'] = 6
    im = sns.clustermap(z,row_linkage=y,col_linkage=y,dendrogram_ratio=(.2, .2),
            cbar_pos=(0.05,0.95,0.2,0.03),xticklabels=[],yticklabels=yticklabels,
            row_colors=colors,col_colors=colors,figsize=(10,10),
            cbar_kws={"orientation": "horizontal","ticks":[0,0.2,0.4,0.6,0.8,1.0]} )
    im.ax_row_dendrogram.set_visible(False)
    if no_cbar:
        im.cax.set_visible(False)
    else:
        pass
        #im.ax_cbar.tick_params(labelsize=10)
        #im.ax_cbar.set_label('cluster frequency')
    if fig_title: im.fig.canvas.set_window_title(fig_title)
    im.reordered_ind = im.dendrogram_row.reordered_ind
    return im
    
