"""
@name: scale.py
@description:

Module to scale arrays and matricies

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""
import scipy.sparse as sp
import numpy as np
import copy

import toolbox.matrix_properties as mp

def sum_to_target(X,target,axis=0):
    assert np.min(target) > 0
    assert axis in [0,1]
    
    target = float(target)
    if axis: 
        X = X.tocsr()
    else:
        X = X.tocsc()
        X = X.T

    X.data = X.data.astype(np.float64)
    xsum = X.sum(1).A1
    xsum[xsum == 0] = 1
    target /= xsum
    
    assert target.shape[0] == X.shape[0]
    
    X.data *= np.repeat(target, np.diff(X.indptr)).reshape(-1)
    X = X.tocoo()
    if not axis: X = X.T
    
    return X

def normalize_to_median(X,axis=0):
    assert axis in [0,1]
    
    if axis: 
        X = X.tocsr()
    else:
        X = X.tocsc()
        X = X.T

    X.data = X.data.astype(np.float64)
    xsum = X.sum(1).A1
    target = np.median(xsum[xsum>0])
    xsum[xsum == 0] = 1
    target /= xsum
    assert target.shape[0] == X.shape[0]
    X.data *= np.repeat(target, np.diff(X.indptr)).reshape(-1)
    X = X.tocoo()
    if not axis: X = X.T
    return X

def size_factor(X,axis=1):
    xsum = mp.axis_counts(X,axis=axis)
    return xsum / np.exp(np.mean(np.log(xsum)))


def log_transform(X):
    if sp.issparse(X):
        X.data = np.log(X.data)
    else:
        return np.log(X + 1)

def standardize(X,axis=0):
    std = X.std(axis=axis)
    if axis == 1: std = std.reshape(-1,1)
    return np.true_divide(X - X.mean(axis=axis,keepdims=True),std)

def minmax(X,axis=0):
    X = np.true_divide(X-X.min(axis=axis,keepdims=True),X.max(axis)-X.min(axis))
    return X 
