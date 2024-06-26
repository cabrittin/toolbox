"""
@name: diffusion.py
@description:
    Module for diffusion classes

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

class Diffusion2D:
    def __init__(self,dx=1,dy=1,D=4.):
        dx2, dy2 = dx*dx, dy*dy
        dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
        self.params = [dx,dy,dx2,dy2,dt,D]
        

    def iter(self,u0,u):
        dx,dy,dx2,dy2,dt,D = self.params
        
        u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
          (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
          + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )

        u0 = u.copy()
        return u0, u
