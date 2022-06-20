import numpy as np
from scipy.signal import argrelmax,argrelmin
import collections
import heapq


def center_warping(warping,t):
    avg=np.zeros(t.shape)
    for x in warping:
        avg+=x/len(warping)
    g=np.zeros(avg.shape)
    dg=np.diff(avg)+10**-3
    g[1:]=np.cumsum(dg)
    g=g/g[-1]
    return g
    
def truncate(x1,t1):
    dx1=np.abs(np.diff(x1))
    b1=np.array([True]+(dx1[:-1]>0).tolist()+[True])  
    x1=x1[b1]
    t1=t1[b1]
    
    maxs=argrelmax(x1)
    mins=argrelmin(x1)     
    idx=np.concatenate((np.array(maxs[0]),np.array(mins[0])))
    idx.sort(kind='mergesort')
    idx=np.concatenate((np.array([0]),idx,np.array([x1.shape[0]-1])))
    return x1[idx],t1[idx]
    

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]