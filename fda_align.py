import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils import *



def precise_align(x1,t1,x2,t2,display=False,to_align=None,to_align_w=None):
    
    f1=interp1d(t1,x1)
    f2=interp1d(t2,x2)
    
    #Truncate
    x1,tt1=truncate(x1,t1)
    x2,tt2=truncate(x2,t2)
        
    #Take derivatives
    dx1=np.diff(x1)
    dx2=np.diff(x2)    
    
    #Define Hueristic  for A*    
    H_cache=-1*np.ones((x1.shape[0],x2.shape[0]))    
    def H(next):
        i,j=next
        if H_cache[i,j]<0:
            par_dx1=dx1[i:]
            par_dx2=dx2[j:]
            H_cache[i,j]=(par_dx1[par_dx1>0].sum()-par_dx2[par_dx2>0].sum())**2+(par_dx1[par_dx1<0].sum()-par_dx2[par_dx2<0].sum())**2
        return H_cache[i,j] 
        
    
    #Define Neighbors and Costs
    def get_neighbors_w_costs(current):
        i,j=current
        ls=[]
        costs=[]
        
        if i==x1.shape[0]-1:
            return [(i,j+1)],[dx2[j]**2]
        if j==x2.shape[0]-1:
            return [(i+1,j)],[dx1[i]**2]
        
        v=dx1[i]*dx2[j]
        if v>0:
            ls+=[(i+1,j+1)]
            costs+=[dx2[j]**2+dx1[i]**2-2*v]
        else:            
            ls+=[(i+1,j)]
            costs+=[dx1[i]**2]
            ls+=[(i,j+1)]
            costs+=[dx2[j]**2]
            
        
        
        return ls,costs
    
    
    #A* Algorithm 
    start = (0,0)
    goal = (dx1.shape[0],dx2.shape[0])

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break
            
        neighbors,segment_costs=get_neighbors_w_costs(current)

        for i in range(0,len(neighbors)):
            next=neighbors[i]
            segment_cost=segment_costs[i]
            new_cost = cost_so_far[current] + segment_cost
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + H(next)
                frontier.put(next, priority)
                came_from[next] = current
                
    #Reconstruct Path
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    n=len(path)
    t=np.linspace(0, 1, num=n, endpoint=True)
    rt1=np.zeros((n))
    rt2=np.zeros((n))  
    for p in range(1,n):
        i,j=path[p]
        rt1[p]=tt1[i]  
        rt2[p]=tt2[j]
    g1= interp1d(t,rt1)
    g2= interp1d(t,rt2)
    

    # Plot aligned functions
    if display:
        plt.plot(t,f1(g1(t)))
        plt.plot(t,f2(g2(t)))
        plt.show()

    #Apply reparams to the functions we want to warp
    if to_align is not None:    
        aligned_list=[]
        for rxn in to_align:
            fn=interp1d(t1,rxn)
            aligned_list+=[fn(g1(t))]
        if to_align_w is not None:
            aligned_warping=[]
            for gn in to_align_w:
                gn = interp1d(t1,gn)
                aligned_warping+=[gn(g1(t))] 
            return f1(g1(t)),f2(g2(t)),g1(t),g2(t),t, aligned_list, aligned_warping
        return  f1(g1(t)),f2(g2(t)),g1(t),g2(t),t, aligned_list
    return f1(g1(t)),f2(g2(t)),g1(t),g2(t),t

   

def precise_group_align(ls):
    f=ls[0]
    rx0=f[0,:]
    t=f[1,:]
    aligned=[]
    warping=[]
    
    for i in range(1,len(ls)):
        print("Matching {}/{}".format(i,len(ls)),end="\r")
        f=ls[i]
        xi=f[0,:]
        ti=f[1,:]

        rx0,rxi,g0,gi,t,aligned,warping=precise_align(rx0,t,xi,ti,to_align=aligned, to_align_w=warping)
        
        aligned+=[rxi]
        warping+=[gi]
    print("Matching Complete" ,end="\r")
    
    aligned=[rx0]+aligned
    warping=[g0]+warping
    return aligned,warping,t
    
def precise_karcher_mean(ls,align=True):
    f=ls[0]
    mean=f[0,:]
    t=f[1,:]
    aligned=[]
    warping=[]
    
    for i in range(1,len(ls)):
        m=1.0/(i+1)
        print("Matching {}/{}".format(i,len(ls)),end="\r")
        f=ls[i]
        xi=f[0,:]
        ti=f[1,:]

        rx0,rxi,g0,gi,t,aligned,warping=precise_align(mean,t,xi,ti,to_align=aligned, to_align_w=warping)
        mean=(1-m)*rx0+m*rxi
        
        if align:
            aligned+=[rxi]
            warping+=[gi]
    
    print("Matching Complete" ,end="\r")
    
    
    return mean, [rx0]+aligned, [g0]+warping,t