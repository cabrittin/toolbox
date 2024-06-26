"""
@name: ace_filter.py
@description:
    Module for filtering values of an array based on the Average Causal Effect (ACE)

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import numpy as np
from itertools import combinations

def encode_onehot(labels) -> np.ndarray:
    """
    Converts list of list with integer labels to a 
    one hot enocded matrix

    Args:
    -----
    labels: list,ndarray (num_objects,)
        List of object integer labels
    
    Returns:
    --------
    ohs : ndarray (num_objects, num_unique_labels
        A binarized one hot encoded array
    """
    udx = sorted(np.unique(labels))
    one_hot = np.zeros((len(labels),len(udx)),dtype=int)
    for (i,u) in enumerate(udx):
        one_hot[np.where(labels==u)[0],i] = 1
    return one_hot

def parse_domains(domains:list,num_levels:int=0):
    """
    Formats a domain list for downstream filtering. Domains is
    a list of one hot encoder matrix column indices. 
    The num_level flag will combine the domains into hierarchical 
    combinations. All hierarchical domains will be outputed as a 
    list of lists.

    Args:
    -----
    domians: list
       List of columns indices for the one hot encoder matrix
    num_level: int, optional (default: 0)
       Sets the the number of domain combinations to consider. 
       By default all possible combinations of domains are 
       considered

    Return:
    -------
    doms : list of lists
        List of domain hierarchies


    Examples:
    > doms = list(range(3))
      #[0, 1, 2]
    > parse_domains(doms)
      #[[0], [1], [2], [0, 1], [0, 2], [1, 2]]
    > parse_domains(doms,num_levels=1)
      #[[0], [1], [2]]
    > parse_domains(doms,num_levels=2)
      #[[0], [1], [2], [0, 1], [0, 2], [1, 2]]
    """
    
    if num_levels == 0: num_levels = len(domains)-1
    doms = []
    for i in range(1,num_levels+1):
        doms += [list(c) for c in combinations(domains,i)]
    return doms

def split_domains(X:np.ndarray,pos_dom:list,neg_dom:list=None)-> (np.ndarray,np.ndarray):
    """
    Splits a one hot encoded array into the postive (X=do(x)) and negative (X=do(~x)) 
    instances

    Args:
    -----
    X : ndarray, (m,n)
      A one hot encoded array
    pos_dom: list
      List of X indices to conider as the postive do(x) instances
    neg_dom: list, optional (default: None)
      List of X indices to condider as the negative do(~x) instances. If not provided,
      the complement of x_pos will be used.

    Examples
    > ohe = np.where(np.random.uniform(0,1,size=(10,2))>0.6,1,0)
    > split_domains(ohx,[0])
      #(array([1, 0, 1, 1, 0]), array([0, 1, 0, 0, 1]))

    """
    xpos = X[:,pos_dom].sum(1)
    xpos[xpos > 0] = 1
    if neg_dom is not None:
        xneg = X[:,neg_dom].sum(1) 
        xneg[xneg > 0] = 1
    else:
        xneg = 1 - xpos
    return xpos,xneg

def precompute_pz(Z:np.ndarray) -> np.ndarray:
    """
    Convenience function to precompute Pr(Z=z_i).
    Assuming that the Z labels are fixed, then precomputing pz can save time.

    Args:
    -----
    Z: ndarray (m,n)
     One hot encoded array of the Z labels
    
    Return:
    -------
    A size n array of precomputed pz probabilities

    """
    return Z.sum(0) / float(Z.sum())


def adjusted_causal_effect(adjusted_formula):
    """
    A decorator function for computing the adjusted causal effect.

    The decorator provides the outer looping structure but the
    adjusted_formula for computing the probability must be provided
    by the user

    Args:
    adjusted_formula: function
     Should accept the following:
     
     adjusted_formulat(Y,X,*args,**kwargs)
    
     Args:
     ----
     Y : ndarray (m,)
       A 1D target array
     X : ndarray (m,)
       A 1D binary array with the do(X=1) labels
    *args:
       Any additional postional arguments
    **kwargs:
       Any additional kwargs
    """
    def inner(Y:np.ndarray,xpos:np.ndarray,xneg:np.ndarray,*args,**kwargs)->np.ndarray:
        """
        Args:
        -----
        Y : ndarray (m,n)
          An array of target variables with m objects (rows) and n variables (cols),
          The average causal effect will be computed for each variable.
        xpos : ndarray (m,)
          A binary array with the positive (do(x=1)) events labeled
        xneg : ndarray (m,)
          A binary array with the negative (do(x=0)) events labeled, should be
          the complement of xpos
        
        *args:
          Additional positional arguments required for the adjusted_formula

        **kwargs:
          Addition kwargs required for the adjusted formulat
    
        Return:
        ------
        ace: ndarray (n,)
           A 1D array of the average causal effect for each variable
        """

        ace = np.zeros(Y.shape[1])  
        for i in range(Y.shape[1]):
            pos = adjusted_formula(Y[:,i],xpos,*args,**kwargs)
            neg = adjusted_formula(Y[:,i],xneg,*args,**kwargs)
            ace[i] = pos - neg
        return ace
    return inner

@adjusted_causal_effect
def zxy_adjusted(
        Y:np.ndarray,
        X:np.ndarray,
        *args,
        pz:np.ndarray=None,
        **kwargs)->np.ndarray:
    """
    Adjusted formuala that assumes the following structural causal model
    
    Z->X, Z->Y, X->Y

    Controls for the effects of Z by fixing X (do(X=1))

    The adjusted probability is computed as

    Pr(Y=1 | do(X=1)) = ∑_i Pr(Y=1| X=1, Z=i)Pr(Z=i)
    
    Args:
    -----
    Y = ndarray (m,)
      A 1D array of the binarized target values
    X: ndarray (m,1)
      A 1D array of the binarized do(X=1) labels.

    *args:
      Position dummy required for the decorator

    pz: ndarray (m,), optional (default: None):
      A 1D array of precompute Pr(Z=i) values. pz[i] = Pr(Z=i)
    
    **kwargs:
      kwargs dummy required for the decorator

    """
    eps = 1e-5
    if pz is None: pz = args[0].sum(0) / float(args[0].sum())
    psum = 0
    for i in range(args[0].shape[1]):
        xz = X*args[0][:,i]
        yxz = Y*xz
        pjoint = yxz.sum() / (xz.sum() + eps)
        psum += pjoint * pz[i]
    return psum


