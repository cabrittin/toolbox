"""
@name:plots.py
@description:

Module for making common plots

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from tqdm import tqdm


class plot_multi_pages(object):
    def __init__(self,ax_dim=(5,5),figsize=(15,15),fout=None,**kwargs):
        self.ax_dim = ax_dim
        self.chunk_size = ax_dim[0]*ax_dim[1]
        self.figsize = figsize
    
    def __call__(self,func,**kwargs):
        def wrapper(data,num_figs,fout=None,ylabel='',xlabel='',**kwargs):
            desc = 'Multi-plot'
            tot = num_figs // self.chunk_size
            for (k,jdx) in tqdm(enumerate(chunks(range(num_figs),self.chunk_size)),
                                total=tot,desc=desc):
                fig,_ax = plt.subplots(self.ax_dim[0],self.ax_dim[1],figsize=(15,15))
                ax = _ax.flatten() 
                for (i,j) in enumerate(jdx): 
                    func(data,index=j,ax=ax[i],**kwargs) 
                    ax[i].tick_params(axis='x',labelsize=6)
                    ax[i].tick_params(axis='y',labelsize=6)

                """Clean up empty plots"""
                if i < self.chunk_size-1:
                    for _i in range(i+1,self.chunk_size):
                        ax[_i].set_xticklabels('')
                        ax[_i].set_yticklabels('')

                for i in range(_ax.shape[0]): _ax[i,0].set_ylabel(ylabel,fontsize=8)
                for i in range(_ax.shape[1]): _ax[-1,i].set_xlabel(xlabel,fontsize=8)
                
                if fout:
                    _fout = fout.replace('.png',f'_{k}.png')

                    plt.savefig(_fout)

                plt.clf()

            return None
        return wrapper

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def viz_scatter(ax,x,y,df,callback=None,**kwargs):
    im = ax.scatter(x,y,**kwargs)
    add_hover(df,callback) 
    return im
    """
    if callback is not None: 
        cursor = mplcursors.cursor(hover=True)
        cursor.connect(
            "add", lambda sel: sel.annotation.set_text(callback(df,sel.target.index))
            )
    return im
    """
def add_hover(df,callback=None):
    if callback is not None: 
        cursor = mplcursors.cursor(hover=True)
        cursor.connect(
            "add", lambda sel: sel.annotation.set_text(callback(df,sel.target.index))
    )
   

def row_of_images(images, show_n=10, image_shape=None):
    """
    Plots rows of images from list of iterables (iterables: list, numpy array
    or torch.Tensor). Also accepts single iterable.
    Randomly selects images in each list element if item count > show_n.

    Args:
        images (iterable or list of iterables)
            single iterable with images, or list of iterables

        show_n (integer)
            maximum number of images per row

        image_shape (tuple or list)
            original shape of image if vectorized form

    Returns:
        Nothing.
    """

    if not isinstance(images, (list, tuple)):
        images = [images]

    for items_idx, items in enumerate(images):

        items = np.array(items)
        if items.ndim == 1:
          items = np.expand_dims(items, axis=0)

        if len(items) > show_n:
          selected = np.random.choice(len(items), show_n, replace=False)
          items = items[selected]

        if image_shape is not None:
          items = items.reshape([-1] + list(image_shape))

        plt.figure(figsize=(len(items) * 1.5, 2))
        for image_idx, image in enumerate(items):

          plt.subplot(1, len(items), image_idx + 1)
          plt.imshow(image, cmap='gray', vmin=image.min(), vmax=image.max())
          plt.axis('off')

          plt.tight_layout()

def xy_lim(x):
    """
    Return arguments for plt.xlim and plt.ylim calculated from minimum
    and maximum of x.

    Args:
        x (list, numpy array or torch.Tensor of floats)
            data to be plotted

    Returns:
        Nothing.
    """

    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)

    x_min = x_min - np.abs(x_max - x_min) * 0.05 - np.finfo(float).eps
    x_max = x_max + np.abs(x_max - x_min) * 0.05 + np.finfo(float).eps

    return [x_min[0], x_max[0]], [x_min[1], x_max[1]]


def scree(data,ax1=None,with_cumsum=False):
    if ax1 is None: fig,ax1 = plt.subplots(1,1,figsize=(3,3))
    ax1.plot(data,'o-',markersize=3)
    ax1.set_ylim([0,data.max()])
    ax1.set_ylabel('$|\lambda_i|$',fontsize=8)
    ax1.set_xlabel('index ($i$)',fontsize=8)
    ax1.tick_params(axis='x',labelsize=6)
    ax1.tick_params(axis='y',labelsize=6)
    if with_cumsum: 
        csum = np.cumsum(data) / data.sum() 
        ax2 = ax1.twinx() 
        ax2.plot(csum,'o-',color='r',markersize=3)
        ax2.set_ylim([0,1]) 
        ax2.set_ylabel('Cumulative variance',fontsize=8) 
        ax2.tick_params(axis='y',labelsize=6)
        
    plt.tight_layout()

def heatmap(ax,Z,title=None,vmin=0,vmax=1,cmap='plasma',**kwargs):
    ax.imshow(Z,vmin=vmin,vmax=vmax,cmap=cmap,**kwargs) 
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None: ax.set_title(title,fontsize=8)

def add_lines(ax,X,Y,**kwargs):
    for i in range(Y.shape[0]):
        c1 = Y[i,:2]
        c2 = X[i,:2]
        xval = [c1[0],c2[0]]
        yval = [c1[1],c2[1]]
        ax.plot(xval,yval,**kwargs)

def to_s2(u):
    """
    Projects 3D coordinates to spherical coordinates (theta, phi) surface of
     unit sphere S2.
     theta: [0, pi]
    phi: [-pi, pi]

    Args:
        u (list, numpy array or torch.Tensor of floats)
        3D coordinates

    Returns:
        Sperical coordinates (theta, phi) on surface of unit sphere S2.
    """

    x, y, z = (u[:, 0], u[:, 1], u[:, 2])
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(x, y)

    return np.array([theta, phi]).T


def to_u3(s):
  """
  Converts from 2D coordinates on surface of unit sphere S2 to 3D coordinates
  (on surface of S2), i.e. (theta, phi) ---> (1, theta, phi).

  Args:
    s (list, numpy array or torch.Tensor of floats)
        2D coordinates on unit sphere S_2

  Returns:
    3D coordinates on surface of unit sphere S_2
  """

  theta, phi = (s[:, 0], s[:, 1])
  x = np.sin(theta) * np.sin(phi)
  y = np.sin(theta) * np.cos(phi)
  z = np.cos(theta)

  return np.array([x, y, z]).T



