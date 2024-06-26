"""
@name: optimize.py
@description:
    
    Various optimizers

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import numpy as np
from tqdm import tqdm

class GradDescent:
    """
    Base optimization class to be inherited
    """
    def __init__(self,cost,theta,args=(),learn_rate=0.1,n_iter=500,tol=1e-6,
            record=False,verbose=False,dtype="float64"):
        """
        Args:
        -----
        cost: a cost function object
            Cost function object     
        w: numpy array,
            Inital start value
        args: tuple, optional
            Tuple of args used for grad function
        learn_rate: float, optional (default 0.1)
            Learing rate
        n_iter: int, optional (default 500)
            Max number of iteration
        tol: fload, optional (default 1e-6)
            Tolerance, programe will exit when abs(xnew - xold) < tol
        record: bool, optional (default False)
            If true, record value during optimization
        verbose: bool, optional (default False)
            If true, display progress
        dtype: str or numpy.dtype, optional (default float64)
            Datatype
        """
        
        if not callable(cost): raise TypeError("'cost' must be callable")
        
        grad_call = getattr(cost, "grad", None)
        if not callable(grad_call): raise TypeError("'cost' must have 'grad()' method")
        
        dtype_ = np.dtype(dtype)
        theta = np.array(theta, dtype=dtype_)
        #x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
        
        learn_rate = np.array(learn_rate, dtype=dtype_)
        if np.any(learn_rate <= 0):
            raise ValueError("'learn_rate' must be greater than zero")
        
        n_iter = int(n_iter)
        if n_iter <= 0: raise ValueError("'n_iter' must be greater than zero")
        
        tol = np.array(tol, dtype=dtype_)
        if np.any(tol <= 0):
            raise ValueError("'tolerance' must be greater than zero")

        self.cost = cost
        self.theta = theta
        #self.x = x
        #self.y = y
        self.args = args
        self.lr = learn_rate
        self.n_iter = n_iter
        self.tol = tol
        self.history = []
        self.record = record
        self.verbose = verbose
        self.dtype_ = dtype_
        self.track_loss = []
        self.exit = False

    def run(self):
        if self.record: self.history.append(self.cost.objective(self.theta,*self.args))
        for i in tqdm(range(self.n_iter),desc='Opt iter',disable= not self.verbose):
            self.t = i
            self.step()
            loss = self.cost(self.theta,*self.args)
            self.track_loss.append(loss)
            if loss < self.tol: break

    def step(self):
        diff = -self.lr * self.cost.grad(self.theta,*self.args)
        self.theta += diff 
        if self.record: self.history.append(self.cost.objective(self.theta,*self.args))
        return self.cost(self.theta,*self.args)  
    

class SGD(GradDescent):
    """ 
    Stochastic gradient descent
    
    Inherits GradDescent
    
    Additionally requires input (x) and target (y), which must be accepted as by
    the cost funciton as cost(w,x,y)

    """
    def __init__(self,*args,batch_size=1,random_state=None,
                decay_rate=0.0,**kwargs):
        super().__init__(*args,**kwargs)
        n_obs = self.x.shape[0]
        if n_obs != self.y.shape[0]:
            raise ValueError("'x' and 'y' lengths do not match")
        rdx = np.arange(self.x.shape[0]) 
        rs = np.random.RandomState(random_state)

        batch_size = int(batch_size)
        if not 0 < batch_size <= n_obs:
            raise ValueError(
                "'batch_size' must be greater than zero and less than "
                "or equal to the number of observations"
            )
       
        decay_rate = np.array(decay_rate, dtype=self.dtype_)
        if np.any(decay_rate < 0) or np.any(decay_rate > 1):
            raise ValueError("'decay_rate' must be between zero and one")

        self.rdx = rdx
        self.rs = rs
        self.n_obs = n_obs
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.diff = 0

    def step(self):
        self.rs.shuffle(self.rdx)
        for start in range(0, self.n_obs, self.batch_size):
            self.batch_step(start) 
            if self.exit: break

    def batch_step(self,start):
        stop = start + self.batch_size
        bdx = self.rdx[start:stop]
        grad = np.array(
                self.cost.grad(self.w,self.x[bdx],self.y[bdx],*self.args), 
                self.dtype_)
        self.update(grad) 
        if self.record: self.history.append(self.w)
    
    def update(self,grad):
        self.diff = self.decay_rate*self.diff + self.learn_rate * grad
        self.w -= self.diff
 
            
class Adagrad(SGD):
    """ 
    Adagrad (adaptive gradient) 
    
    Inherits SGD
    
    Additionally requires input (x) and target (y), which must be accepted as by
    the cost funciton as cost(w,x,y)

    """
    def __init__(self,cost,w,x,y,**kwargs):
        super().__init__(cost,w,x,y,**kwargs)
        self.gradient_sum = np.zeros(w.shape[0])
        self._eps = 1e-8
        
    def update(self,grad):
        self.gradient_sum += grad ** 2
        grad_update = grad / np.sqrt(self.gradient_sum + self._eps)
        if np.all(grad_update) > 1: grad = grad_update
        self.diff = - self.learn_rate * grad
        self.w += self.diff
 
class Adadelta(SGD):
    """ 
    Adagrad (adaptive gradient) 
    
    Inherits SGD
    
    Additionally requires input (x) and target (y), which must be accepted as by
    the cost funciton as cost(w,x,y)

    """
    def __init__(self,*args,rho=0.8,**kwargs):
        super().__init__(*args,**kwargs)
        rho = np.array(rho, dtype=self.dtype_)
        if np.any(rho <= 0):
            raise ValueError("'rho' must be greater than zero")
 
        self.rho = rho
        self.eps = 1e-5
        
        # Running average of gradient
        self.s =  np.zeros(self.w.shape[0])
        # Running average of rescaled gradient
        self.delta =  np.zeros(self.w.shape[0])
    
    def update(self,grad):
        self.s = self.rho*self.s + (1-self.rho)*(grad**2)
        self.diff = np.sqrt(self.delta + self.eps) / np.sqrt(self.s+self.eps) * grad
        self.delta = self.rho*self.delta + (1-self.rho)*(self.diff**2)
        self.w -= self.diff

class Adam(SGD):
    """ 
    Adagrad (adaptive gradient) 
    
    Inherits SGD
    
    Additionally requires input (x) and target (y), which must be accepted as by
    the cost funciton as cost(w,x,y)

    """
    def __init__(self,*args,beta1=0.9,beta2=0.9,**kwargs):
        super().__init__(*args,**kwargs)
        beta1 = np.array(beta1, dtype=self.dtype_)
        if np.any(beta1 <= 0):
            raise ValueError("'beta1' must be greater than zero")
        
        beta2 = np.array(beta2, dtype=self.dtype_)
        if np.any(beta2 <= 0):
            raise ValueError("'beta2' must be greater than zero")
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-5
        
        # Running average of gradient
        self.s =  np.zeros(self.w.shape[0])
        # Running average of rescaled gradient
        self.v =  np.zeros(self.w.shape[0])
    
    def update(self,grad):
        self.v = self.beta1*self.v + (1-self.beta1)*grad
        self.s = self.beta2*self.s + (1-self.beta2)*(grad**2)
        v_bias_cor = self.v / (1. - self.beta1**(self.t+1))
        s_bias_cor = self.s / (1. - self.beta2**(self.t+1))
        self.diff = self.learn_rate * v_bias_cor / (np.sqrt(s_bias_cor) + self.eps)
        self.w -= self.diff
