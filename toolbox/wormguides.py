"""
@name: wormguides.py
@description:

Module to handle wormguides related data

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import networkx as nx
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import defaultdict,namedtuple 
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import editdistance

from pycsvparser import read

class Cell:
    def __init__(self,name):
        self.name = name
        self.pos = []
        self.time = []
        self.parent_idx = None
        self.parent = None
        self.mean_pos = None
        self.delta_t = 0
        self.tmin = 0
        self.tmax = 0
        self.tmean = 0
        self.tpmap = defaultdict(int)
        self.tdx = 0
        self.ap = []
        self.ap_segment = []

    def format_measures(self):
        self.pos = np.array(self.pos)
        self.mean_pos = np.mean(self.pos,axis=0)
        self.time = np.array(self.time)
        self.tmin = self.time.min()
        self.tmax = self.time.max()
        self.tmean = np.mean(self.time)
        self.delta_t = self.tmax - self.tmin

    def add_position(self,time,pos):
        self.pos.append(pos)
        self.time.append(time)
        self.tpmap[time] = self.tdx
        self.tdx += 1

    def get_position(self,time):
        tdx = self.tpmap[time]
        return self.pos[tdx,:]



class Embryo:
    def __init__(self):
        self.Cell = namedtuple("Cell","idx name time parent_idx")
        self.data = {}
        self.cells = {}
        self.time_rec = defaultdict(list)

    def load(self,emb_file):
        self.fin = emb_file
        count = 0
        with open(emb_file, 'r') as f:
            for line in f: count += 1

        self.pos = np.zeros((count,3))
        
        with open(emb_file,'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                self.add_data(row)

    def add_data(self,row):
        """
        row format: [index,time,cell name, parent index, x, y, z]
        """
        idx = int(row[0]) - 1
        time = int(row[1])
        cell = row[2]
        self.time_rec[time].append(idx)
        parent_idx = int(row[3]) - 1
        pos = np.array([x for x in map(float,row[4:])])
        self.pos[idx] = pos
        self.data[idx] = self.Cell(idx=idx,name=cell,time=time,parent_idx=parent_idx)
        if cell not in self.cells: self.cells[cell] = {}
        if time not in self.cells[cell]: self.cells[cell][time] = idx
        

    def generate_lineage(self):
        self.L = nx.Graph()
        for (k,v) in tqdm(self.data.items(),desc="Lineage graph: "):
            if v.parent_idx < 0: continue
            parent = self.data[v.parent_idx].name
            if parent == v.name: continue
            times = [int(t) for t in self.cells[v.name].keys()]
            tmin = min(times)
            tmax = max(times)
            delta_t = tmax - tmin
            self.L.add_edge(parent,v.name,delta_t=delta_t,tmin=tmin,tmax=tmax)

    def generate_space(self,ap_file,k_neigh=6):
        self.ap_file = ap_file
        self.S = nx.Graph()
        for (idx,d) in self.data.items():
            self.S.add_node(idx,idx=d.idx,name=d.name,parent=d.parent_idx)
            if d.parent_idx > -1: self.S.add_edge(idx,d.parent_idx)
        with open(self.ap_file,'r') as f:
            self.ap_segment = f.readline().rstrip().split(',')
        self.dist = []
        for (time,indices) in tqdm(self.time_rec.items(),desc="Spatial graph: "):
            pos = np.zeros((len(indices),3))
            for (i,idx) in enumerate(indices): 
                pos[i,:] = self.pos[idx,:]
            _k = min(len(indices)-1,k_neigh)  
            A = kneighbors_graph(pos,_k).tocoo()
            for i,j in zip(A.row,A.col): self.S.add_edge(indices[i],indices[j])
            if time < 330:
                self.compute_early_ap(indices,time,pos)
            else:
                self.compute_late_ap(indices,time,pos)
        
        self.add_parent_neighbors()
        #self.prune_spatial_ap()   
    
    def add_parent_neighbors(self):
        nodes = self.S.number_of_nodes() - 1
        print(self.S.number_of_edges())
        for u in tqdm(reversed(range(nodes)),desc="Adding parent neighbors"):
            pdx = self.S.nodes[u]['parent'] 
            if pdx < 0: continue
            for v in self.S.neighbors(pdx):
                self.S.add_edge(u,v)
        print(self.S.number_of_edges())

    def prune_spatial_ap(self):
        """Prune S based on AP differences"""
        eprune = []
        ap_max = 0.10
        for (u,v) in self.S.edges():
            ap_diff = abs(self.S.nodes[u]['ap'] - self.S.nodes[v]['ap'])
            self.dist.append(ap_diff)
            if ap_diff > ap_max: eprune.append((u,v))
        print(self.S.number_of_edges())
        self.S.remove_edges_from(eprune)
        print(self.S.number_of_edges())

    def compute_early_ap(self,indices,time,pos):
        min_x = pos[:,0].min()
        max_x = pos[:,0].max()
        dx = max_x - min_x
        for (i,idx) in enumerate(indices): 
            _pos = self.pos[idx,:]
            ap = (_pos[0] - min_x) / dx
            self.S.nodes[idx]['ap'] = ap
    
    def compute_late_ap(self,indices,time,pos):
        lines = []
        ap_length = 0
        
        """Construct AP segments"""
        for i in range(1,len(self.ap_segment)):
            c1 = self.ap_segment[i-1]
            c2 = self.ap_segment[i]
            idx1 = self.cells[c1][time]
            idx2 = self.cells[c2][time]
            x1 = self.pos[idx1,:]
            x2 = self.pos[idx2,:]
            lines.append(Line(x1,x2))
        
        """Project cell positions onto AP"""
        cum_length = np.cumsum([l.length for l in lines])
        cum_length = np.concatenate((np.array([0]),cum_length),axis=0) 
        ap = []
        for (i,idx) in enumerate(indices):
            ldist = [l.distance_to(pos[i,:]) for l in lines]
            lmin = np.argmin(ldist)
            proj = lines[lmin].projection(pos[i,:])
            _ap = (cum_length[lmin] + proj) / cum_length[-1]
            ap.append(_ap)

        """Scale AP"""
        ap = np.array(ap)
        ap /= np.max(ap)
        for (i,idx) in enumerate(indices):
            self.S.nodes[idx]['ap'] = ap[i]

    def save_lineage(self,fout):
        nx.write_graphml(self.L,fout)
   
    def save_space(self,fout):
        nx.write_graphml(self.S,fout)


class Line:
    def __init__(self,x1,x2):
        self.x1 = x1
        self.x2 = x2
        self.length = np.linalg.norm(x2-x1)
        self.unit = (x2-x1) / self.length

    def distance_to(self,x0):
        cross = np.cross((self.x1-x0),(self.x2-x0))
        return np.linalg.norm(cross) / self.length

    def projection(self,x0):
        proj = np.dot((x0 - self.x1),self.unit)
        #r = self.x1 + proj*self.unit
        return proj


class LineageNameMatch:
    def __init__(self):
        self.keys = ['AB','MS','D','C','E']

    def build_dict(self,source):
        self.store = dict([(k,defaultdict(list)) for k in self.keys])
        for s in source:
            size = len(s)
            for k in self.keys:
                if not k in s: continue
                self.store[k][size].append(s)

    def find_match(self,name):
        match_found = False
        for k in self.keys:
            if k not in name: continue
            size = len(name)
            ed = np.array([editdistance.eval(name,w) for w in self.store[k][size]])
            idx = np.where(ed == ed.min())[0]
            name = [self.store[k][size][i] for i in idx]
            match_found = True
        return name,match_found
 
