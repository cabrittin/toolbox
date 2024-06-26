"""
@name:
@description:


@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import argparse
import numpy as np
import time


from toolbox.bigdata.io import *

def t_to_npz():
    X = np.random.random(size=(10000,100000))
    fout = 'data/t_bd.npz'
    dout = 'data/'
    chunk_size = 10000
    
    print('Data intialized...')
    # Serial performance
    t0 = time.time()
    to_npz_serial(fout,X,chunk_size) 
    print('Time to serial write %1.4f sec'%(time.time()-t0))
    t0 = time.time()
    X2 = from_npz_serial(dout) 
    print('Time to serial read %1.4f sec'%(time.time()-t0))
    print(f'Arrays equal: {np.array_equal(X,X2)}')

    # Parallel performance 
    t0 = time.time()
    to_npz(fout,X,chunk_size) 
    print('Time to parallel write %1.4f sec'%(time.time()-t0))
    t0 = time.time()
    X2 = from_npz(dout) 
    print('Time to parallel read %1.4f sec'%(time.time()-t0))
    print(f'Arrays equal: {np.array_equal(X,X2)}')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode',
                        action = 'store',
                        help = 'Mode to run')
 
    params = parser.parse_args()

    
    eval(params.mode + '()')
