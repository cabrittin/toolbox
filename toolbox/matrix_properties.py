"""
@name: matrix_properies.py
@description:

Module for getting matrix properies

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import warnings
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from time import process_time

ATTR = ['tocsc','tocsr']

def axis_counts(X,axis=0):
    assert axis in [0,1] 
    return getattr(X,ATTR[axis])().sum(axis).A1 

def axis_elements(X,axis=0):
    assert axis in [0,1]
    return np.diff(getattr(X,ATTR[axis])().indptr)

def counts_in_range(counts, min_counts=-1, max_counts=1e15):
    assert max_counts > min_counts
    counts[ (counts < min_counts) | (counts > max_counts) ] = 0
    return np.where(counts > 0)

def axis_mean(X,axis=0,skip_zeros=False):
    assert axis in [0,1]
    sums = getattr(X,ATTR[axis])().sum(axis).A1
    if skip_zeros:
        counts = np.diff(getattr(X,ATTR[axis])().indptr)
        counts[counts < 1] = 1
    else:
        counts = np.ones(X.shape[~axis])*X.shape[axis]
    return sums / counts

def axis_mean_var(X,axis=0,skip_zeros=False): 
    assert axis in [0,1]
    if X.dtype != np.float64:
        warnings.warn(f"Converted from dtype {X.dtype} to np.float64")
        #Amazingly converting dtype of .data directly is an order of magnitude faster than X.astype()
        X.data = X.data.astype(np.float64)

    X = getattr(X,ATTR[axis])()
    sums = X.sum(axis).A1
    sums_sq = X.multiply(X).sum(axis).A1
    if skip_zeros:
        counts = np.diff(getattr(X,ATTR[axis])().indptr)
        counts[counts < 1] = 1
    else:
        counts = np.ones(X.shape[~axis])*X.shape[axis]
    mean = sums / counts
    mean_sq = sums_sq / counts
    var = mean_sq - mean**2
    return mean,var

def axis_clip_value(X,clip_val,axis=0):
    """
    Values greater than clip_val are set to clip_val.
    """
    assert axis in [0,1]
    if axis: X = X.T
    if np.isscalar(clip_val): clip_val = clip_val * np.ones(X.shape[1])

    ##Verified to perform as expected -- CB: 2022-04-29
    Xc = X.tocsr().copy()
    mask = Xc.data > clip_val[Xc.indices]
    Xc.data[mask] = clip_val[Xc.indices[mask]]

    if axis: Xc = Xc.T
    return Xc

def var_of_user_standardized_values(X,std,mean=None,axis=0):
    """
    Computes the mean of user standardized values. The wishes to standandize
    X with std, this function returns the variances of these standardized 
    values.

    If mean is provided, then a balance between mean and mean of x is returned.
    """
    assert axis in [0,1]
    if axis: X = X.T
    if np.isscalar(std): std = std * np.ones(X.shape[1])
    
    N = X.shape[0]
    sum_xsq = np.array(X.power(2).sum(axis=0))
    sum_x = np.array(X.sum(axis=0))
    
    if mean is not None:
        ssq = (N * np.square(mean)) + sum_xsq - 2 * sum_x * mean
    else:
        ssq = sum_xsq + (sum_x / N)

    return (ssq / ((N - 1) * np.square(std))).reshape(-1)
