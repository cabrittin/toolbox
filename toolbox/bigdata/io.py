"""
@name: io.py
@description:
    Module for reading/writing large datasets

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import os
import multiprocessing_on_dill as mp
import numpy as np

def chunk_range(max_val,chunk_size):
    for i in range(0,max_val,chunk_size):
        yield i,min(i+chunk_size,max_val)

def search_dir(din):
    for filename in os.listdir(din):
        f = os.path.join(din, filename)
        if os.path.isfile(f): 
            yield f

def mp_writer(fout,data,outer_shape,start,end,is_col):
    np.savez_compressed(fout,data=data,outer_shape=outer_shape,
                            start=start,end=end,is_col=is_col)

def mp_reader(fin):
    with np.load(fin,allow_pickle=True) as data:
        return dict([(k,v) for (k,v) in data.items()])

def to_npz(fout,X,chunk_size,is_col=1):
    if is_col == 0: X = X.T
    procs = []
    for idx,(i0,i1) in enumerate(chunk_range(X.shape[1],chunk_size)):
        _fout = fout.replace('.npz',f'_{idx}.npz')
        proc = mp.Process(target=mp_writer, 
                            args=(_fout,X[:,i0:i1],X.shape,i0,i1,is_col))
        procs.append(proc)
        proc.start()
    
    for proc in procs: proc.join() 

def to_npz_serial(fout,X,chunk_size,is_col=1):
    if is_col == 0: X = X.T
    procs = []
    for idx,(i0,i1) in enumerate(chunk_range(X.shape[1],chunk_size)):
        _fout = fout.replace('.npz',f'_{idx}.npz')
        mp_writer(_fout,X[:,i0:i1],X.shape,i0,i1,is_col)

def from_npz(din):
    with mp.Pool() as pool:
        results = pool.map(mp_reader,search_dir(din))
        X = np.zeros(results[0]['outer_shape'])
        for r in results: X[:,r['start']:r['end']] = r['data']
        if r['is_col'] == 0: X = X.T
        return X

def from_npz_serial(din):
    X = None
    for fname in search_dir(din):
        Xs = np.load(fname)
        if X is None: X = np.zeros(Xs['outer_shape'])
        X[:,Xs['start']:Xs['end']] = Xs['data']
    
    if Xs['is_col'] == 0: X = X.T
    return X


