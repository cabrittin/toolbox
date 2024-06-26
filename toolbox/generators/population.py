"""
@name: population_generator.py
@description:
    Contains class for generating perturbed populations

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import networkx as nx
import numpy as np


class PopulationGenerator:
    def __init__(self,graphs,weight='weight',lower_log_thresh=None,noise=None):
        self.return_as_list = False 
        if isinstance(graphs,list):
            self.dbs = graphs
            self.return_as_list = True
        else: 
            self.dbs = [graphs]
        self.weight=weight
        self.lower_log_thresh = lower_log_thresh
        self.set_log_scale()
        self.noise=noise
        #self.rlmap = self.build_rlmap()
    
    def set_log_scale(self,lower_log_thresh=4):
        w = [] 
        for h in self.dbs:
            w += [_w for (u,v,_w) in h.edges(data=self.weight)]
        
        w = np.log(np.array(w))
        if isinstance(self.lower_log_thresh,float):
            idx = np.where(w > lower_log_thresh)
            w = w[idx]

        self.log_scale =  np.std(w)
        
    
    def pop_population(self,noise=None):
        if noise is None: noise = self.noise
        P = [self.perturb_graph(G,noise) for G in self.dbs]
        if self.return_as_list:
            return P
        else:
            return P[0]
    
    def perturb_graph(self,G,noise):
        P = G.__class__()
        for (u,v,w) in G.edges(data=self.weight):
            w = w*np.exp(np.random.normal(scale=noise)*self.log_scale)
            P.add_edge(u,v,weight=w)
        return P

