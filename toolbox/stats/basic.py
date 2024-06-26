"""
@name: basic.py
@description:

Part of the toolbox.stats package. Includes basic stats computations


@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import scipy.stats
import numpy as np

def ecdf(data,reverse=False,**kwargs):
    n = len(data)
    x = np.sort(data)
    y = np.arange(n)/float(n)
    if reverse: y = 1 - y
    return x,y

def z_score(E,R,axis=0,std_eps=0.01,continuity_correction=False):
    #if continuity_correction: E -= 0.5
    mu = R.mean(axis)
    std = R.std(axis)
    print(E)
    print(mu)
    print(std)
    std[std == 0] = std_eps
    z = np.divide(E - mu, std)
    return z

def p_value(z,one_sided=True):
    if one_sided: 
        pval = scipy.stats.norm.sf(abs(z))
    else:
        pval = scipy.stats.norm.sf(abs(z))*2
    return pval

def holm_bonferroni(pvals,alpha=0.05):
    n = len(pvals)
    rank = np.arange(n) + 1
    hb = alpha / (n - rank + 1)
    idx = np.argsort(pvals)
    pbool = pvals[idx] < hb
    pbool[1:] = np.multiply(pbool[:-1],pbool[1:])
    _pbool = np.zeros(len(pbool),dtype=np.uint)
    _pbool[idx] = pbool
    return pbool

def adjusted_pval(pval):
    """
    Computes adjusted pvals using the 
    Benjamini-Hochberg procedure.
    
    Input:
         pval: N array of pvals

    Output:
         adjpval = float value
    """
    pval.sort()
    N = float(pval.shape[0])
    #for i in range(N):
    #    pval[i] = min(N*pval[i]/(i+1),1)
    idx = np.arange(N) + 1
    pval = N*pval/idx
    pval[pval > 1] == 1
    #print(pval)
    adjpval = np.min(pval)
    return adjpval

def zscore_binarize(X:np.ndarray, zscore:float=0, axis:int=0) -> np.ndarray:
    """
    Binarized an array based on column z-score
    
    Args:
    -----
    X : ndarray (m,n)
      Array of float values

    zscore: float, optional (default: 0)
      Z-score threshold. Values less than z-score are set to 0, while
      the remaining values are set to 1
   
    axis: int, optional (default: 0)
        Axis along which to apply zscore threshold

    Return:
    --------
    Y : ndarray (m,n)
      The binarized array

    """
    assert axis in [0,1], "Axis must be in {0,1}"
    eps = 1e-5 
    if axis: X = X.T 
    X = (X - X.mean(0)) / (X.std(0) + eps) 
    X = np.where(X<zscore,0,1)
    if axis: X = X.T
    return X


