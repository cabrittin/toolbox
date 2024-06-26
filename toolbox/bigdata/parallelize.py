"""
@name: parallelize.py
@description:
    
    Module for parallelizing functions

    These are meant to be generic functions and may not be appropriate for all 
    use cases.


@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

from multiprocessing import shared_memory
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
import concurrent
import numpy as np
from functools import wraps

def chunk_range(max_range,chunk_size):
    for i in range(0,max_range,chunk_size):
        yield i,min(max_range,i+chunk_size)

def parallelize_range(max_range=None,chunk_size=None,n_jobs=1,
                        func=None,args=(),**kwargs):
    """
    Spins pool of processes by chunking a range and passing the endpoints 
    of each chunk to child processes

    """
    futures = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for idx,(start,end) in enumerate(chunk_range(max_range,chunk_size)):
            futures.append(executor.submit(func,*args,index=idx,
                                            start=start,end=end,**kwargs))
        futures, _ = concurrent.futures.wait(futures)
    return futures

def create_shared_memory_nparray(data,name):
    d_size = np.dtype(data.dtype).itemsize * np.prod(data.shape)
    shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    dst = np.ndarray(shape=data.shape, dtype=data.dtype, buffer=shm.buf)
    dst[:] = data[:]
    return shm

def release_shared(name):
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()  

def _test_parallelize_range():
    y = np.arange(1000)
    yshm = create_shared_memory_nparray(y,'y') 
 
    futures = parallelize_range(max_range=1000,chunk_size=200,n_jobs=5,func=_tp_test)
    release_shared('y')
    
    result = np.sum([f.result() for f in futures])
    print(f'Parallel sum: {result}')
    print(f'Correct sum: {np.sum(y)}')


def _tp(func):
    """
    Demonstrates how to use decorator function with @wraps
    """
    @wraps(func)
    def inner(*args,index=None,start=None,end=None,**kwargs):
        return func(*args,index=index,start=start,end=end,**kwargs)
    return inner

@_tp
def _tp_test(*args,index=None,start=None,end=None,**kwargs):
    shm = shared_memory.SharedMemory(name='y') 
    y = np.ndarray((1000,),dtype=np.int64,buffer=shm.buf)
    return np.sum(y[start:end])


