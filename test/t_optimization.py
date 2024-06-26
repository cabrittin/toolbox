"""
@name: t_optimization.py
@description:
    
    Test optimization

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#from toolbox.ml.optimize import GradDescent as opt
#from toolbox.ml.optimize import SGD as opt
#from toolbox.ml.optimize import Adagrad as opt
#from toolbox.ml.optimize import Adadelta as opt
#from toolbox.ml.optimize import Adam as opt
from toolbox import plots

class TestLoss:
    def __call__(self,x,*args):
        return (x**2).sum(0)
    def grad(self,x,*args):
        return 2*x
    def objective(self,x,*args):
        return x**2

def surface_plot(*args,fig=None):
    r_min, r_max = -1.0, 1.0
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    C = TestLoss()

    x, y = np.meshgrid(*args, yaxis)
    if fig is None: fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, C(np.asarray([x,y])), cmap='jet')
    plt.show()

def contour_plot(*args,ax=None):
    bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
    xaxis = np.arange(bounds[0,0], bounds[0,1], 0.1)
    yaxis = np.arange(bounds[1,0], bounds[1,1], 0.1)
    C = TestLoss()
    x, y = np.meshgrid(xaxis, yaxis)
    
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.contourf(x, y, C(np.asarray([x,y])), levels=50, cmap='jet',zorder=0)
    plt.show()

def plot_loss(loss,ax=None):
    if not None: fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(loss)
    ax.set_ylabel('Test loss',fontsize=8)
    ax.set_xlabel('Iteration',fontsize=8)

def run_optimize(params):
    from toolbox.ml.optimize import GradDescent as opt
    
    bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
    x0 = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    cost = TestLoss()
    
    gd = opt(cost,x0,learn_rate=0.01,n_iter=1000,tol=1e-6,record=True,verbose=True)
    gd.run()
    
    plot_loss(gd.track_loss) 
    
    soln = np.asarray(gd.history)
    
    bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
    xaxis = np.arange(bounds[0,0], bounds[0,1], 0.1)
    yaxis = np.arange(bounds[1,0], bounds[1,1], 0.1)
    C = TestLoss()
    x, y = np.meshgrid(xaxis, yaxis)

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.contourf(x, y, cost(np.asarray([x,y])), levels=50, cmap='jet',zorder=0)
    ax.plot(soln[:,0],soln[:,1],'.-',color='w',zorder=3) 
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    plt.show()

def test_opt(params):
    x0 = 0
    args = () 
    #f = TestLoss1()
    #f = TestLoss2()
    f = TestLoss4()
    x = np.array([5, 15, 25, 35, 45, 55])
    y = np.array([5, 20, 14, 32, 22, 38])
    x0 = np.array([0.5,0.5])
    x0 = np.random.normal(size=2)
    args = (x,y)
    #x = np.arange(-12,13)
    gd = opt(f,x0,x,y,learn_rate=0.001,n_iter=1000,tol=1e-6,
            record=False,batch_size=4,decay_rate=0.8,verbose=True)
    #gd = opt(f,x0,args=(x,y),learn_rate=0.0008,n_iter=100000,record=False,verbose=True)
    gd.run()
    print(gd.w)
    
    #x = np.linspace(-4,4,100)
    """
    xrec = np.array(gd.history)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(x,f(x),linewidth=1)
    ax.plot(xrec,f(xrec),'s-',color='r')
    plt.show()
    """
    print(gd.track_loss[-1])
    print('here',f.grad(gd.w,x,y))
    print(gd.w[0]+gd.w[1]*x,y)
    plt.plot(gd.track_loss)
    plt.show()


class TestLoss1:
    def __call__(self,x,*args):
        return x**2
    def grad(self,x,*args):
        return 2*x

class TestLoss2:
    def __call__(self,x,*args):
        return  x**4 - 5*(x**2) - 3*x
    def grad(self,x,*args):
        return 4 * x**3 - 10 * x - 3

class TestLoss3:
    def __call__(self,b,x,y):
        return  np.dot(b,x) - y
    def grad(self,b,x,y):
        res = self.__call__(b,x,y)
        return np.dot(res,x)

class TestLoss4:
    def __call__(self,b,x,y):
        return  b[0] + b[1] * x - y
    def grad(self,b,x,y):
        res = self.__call__(b,x,y)
        return np.array([res.mean(),(res*x).mean()])

    def loss(self,b,x,y):
        return ((self.__call__(b,x,y))**2).mean()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('mode',
                        action = 'store',
                        help = 'Mode to run')
    
    params = parser.parse_args()
    
    eval(params.mode + '(params)')
