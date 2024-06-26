"""
@name: dr.py
@description:
    Module for diminsionality reduction

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""
import numpy as np

def classic_mds(D,n_components=2):
    from numpy.linalg import svd
    n = D.shape[0]
    C = np.eye(n) - np.ones((n,n))/float(n)
    B = -0.5 * C @ D @ C
    U,S,V = svd(B,full_matrices=True)
    
    eigs = S**2 / (n-1)
    
    proj = U[:,:n_components] @ np.diag(S[:n_components]) / np.sqrt(n-1)
    return proj,eigs


def test_mds(D,n_components=2):
    from numpy.linalg import eigh,eig
    n = D.shape[0]
    C = np.eye(n) - np.ones((n,n))/float(n)
    B = -0.5 * C @ D @ C
    print(np.allclose(B,B.T)) 
    eigs,proj = eigh(B)
    print(np.linalg.eigvalsh(B))
    print(eigs)
    return proj,eigs


