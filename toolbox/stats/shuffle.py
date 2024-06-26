"""
@name: shuffle
@description:

Module for doing stats on shuffled emprical data

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import numpy as np
from tqdm import tqdm

def shuffle_callback(X,num_iters,container,callback,col_ind=False,**kwargs):
    """
    Randomly shuffles the rows of 2D array X for num_iters. At each iteration,
    the the shuffled array is passed to the callback function.
    
    Note: Only rows are shuffled. If you wish to shuffle the columns, then pass X.T.
    But then expect X.T to be passed to callback, where callback will need to
    handle accordingly.

    Parameters:
    -----------
    X : 2D array
    num_iters: int
        Number of random shuffles to perform
    container: user defined
        Some container that will hold the output of the call back function.
    callback: function
        Function that acts on the shuffled array. Must take the following 
        parameters in this order.
        callback parameters:
        ---------------------
        _iter: int
            The current shuffle interation.
        X : 2D array
        container: user defined
        **kwargs: any optional keyword arguments
    """
    for _iter in tqdm(range(num_iters),desc='Shuffle callback:'):
        if col_ind:
            idx = np.random.rand(*X.shape).argsort(0)
            X = X[idx, np.arange(X.shape[1])]
        else: 
            np.random.shuffle(X)
        callback(_iter,X,container,**kwargs)

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """ 
    b = a.swapaxes(axis, -1)
    n = a.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    b = b[..., idx]
    return b.swapaxes(axis, -1)


def edge_switch(G,iters=1000,params=None):
    """
    Randomly switch edges in graph G (igraph). Do iters random switches.
    Preserves in degree and out degree.

    Parameters
    ----------
    G : iGraph
     Graph in which edges will be randomly switched
    iters : int (default 1000)
     Number of edge switches to perform 
    params: namedtuple (default None)
     Passes parameters using nametuple

    """
    
    if params:
        iters = int(params.iters)

    N = G.ecount() - 2
    idx = 0
    while idx < iters:
        i = randint(0,N)
        j = i
        while j == i: j = randint(0,N)
        ei = G.es[i]
        ej = G.es[j]
        u1 = ei.source
        v1 = ei.target
        u2 = ej.source
        v2 = ej.target
        iattr = ei.attributes()
        jattr = ej.attributes()
        cond1 = len(set([u1,v1,u2,v2])) == 4
        cond2 = not G.are_connected(u1,v2) 
        cond3 = not G.are_connected(u2,v1) 
        if cond1 and cond2 and cond3: 
            G.delete_edges([i,j])
            G.add_edge(u1,v2,**iattr)
            G.add_edge(u2,v1,**jattr)
            idx += 1
