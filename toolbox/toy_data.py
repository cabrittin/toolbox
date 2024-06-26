"""
@name: toy_data.py
@description:

    Module to generate synthetic ("toy") data

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""


import numpy as np
from random import shuffle

def chunk_array(row_dim,col_dim,num_chunks,row_shuffle=False,col_shuffle=False):
    row_split = np.array_split(np.arange(row_dim),num_chunks)
    col_split = np.array_split(np.arange(col_dim),num_chunks)
    if row_shuffle: random.shuffle(row_split)
    if col_shuffle: random.shuffle(col_split)

    for (i,j) in zip(row_split,col_split):
        yield ((i[0],i[-1]+1),(j[0],j[-1]+1))

def add_white_noise(X,mu=0,sigma=0):
    N,M = X.shape
    X += sigma*np.random.rand(N,M) + mu
