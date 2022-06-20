import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils import *



def precise_align(x1,t1,x2,t2,display=False,to_align=None,to_align_w=None):
    
    f1=interp1d(t1,x1,axis=0)
    f2=interp1d(t2,x2,axis=0)
        
    #Take derivatives
    dx1=np.diff(x1,axis=0)
    dx2=np.diff(x2,axis=0)    
    
    Omega=np.einsum('ji,ki-> jk', dx1, dx2)   
    Q1=(dx1**2).sum(axis=1)
    Q2=(dx2**2).sum(axis=1)
    Q=np.add.outer(Q1,Q2)
    Q=Q-2*Omega
    Q[Omega<0]=np.inf 
    
    #Define Neighbors and Costs
    def get_neighbors_w_costs(current):
        i,j=current
        
        ls=[]
        costs=[]
        if i<dx1.shape[0]:
            ls+=[(i+1,j)]
            costs+=[(dx1[i]**2).sum()]
        else:
            return [(dx1.shape[0],dx2.shape[0])],[(dx2[j:]**2).sum()]
        if j<dx2.shape[0]:
            ls+=[(i,j+1)]
            costs+=[(dx2[j]**2).sum()]
        else:
            return [(dx1.shape[0],dx2.shape[0])], [(dx1[i:]**2).sum()]
        
        if Q[i,j]!=np.inf:
            ls+=[(i+1,j+1)]
            costs+=[Q[i,j]]
        else:
            return ls,costs
            
        
        ## All of the work.jpg
        for k in range(i+1, x1.shape[0]):
            for l in range(j+1,x2.shape[0]):
                if Q[k-1,l-1]==np.inf:
                    break
                    
                m = (t2[l]-t2[j])/(t1[k]-t1[i])
                txadd=(t2[range(j,l+1)]-t2[j])/m
                tx=np.sort(np.unique(np.concatenate([t1[range(i,k)]-t1[i],txadd],axis=0)))
                ty=tx*m+t2[j]
                tx=tx+t1[i]
                
                bx=i
                by=j
                cost = Q[i,j]
                for n in range(1,tx.shape[0]-1):
                    if tx[n]==t1[bx+1] and ty[n]!=t2[by+1]:
                        bx=bx+1    
                        cost+= Q[bx,by]                    
                    elif tx[n]!=t1[bx+1] and ty[n]==t2[by+1]:                    
                        by=by+1
                        cost+= Q[bx,by]
                    else:
                        cost=np.inf
                        break                        
                    if cost==np.inf:
                        break
                if cost!=np.inf:
                    ls+=[(k,l)]
                    costs+=[cost]
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
                frontier.put(next, new_cost)
                came_from[next] = current
                
    #Reconstruct Path
    current = goal
    path = [[t1[-1],t2[-1]]]
    while current != start:
        previous=current
        current = came_from[current]
        i,j=current
        k,l=previous
        if k>i and l>j:
            m = (t2[l]-t2[j])/(t1[k]-t1[i])
            txvar=(t2[range(j,l+1)]-t2[j])/m
            txs=np.sort(np.unique(np.concatenate([t1[range(i,k)]-t1[i],txvar],axis=0)))
            tys=txs*m+t2[j]
            txs=txs+t1[i]
            for n in range(txs.shape[0]-1,-1,-1):
                path.append([txs[n],tys[n]])    
        elif k==i:
            for n in range(l,j-1,-1):
                path.append([t1[i],t2[n]])
        elif j==l:
            for n in range(k,i-1,-1):
                path.append([t1[n],t2[j]])
            
    path.append([0.0,0.0])
    path.reverse()
    
    n=len(path)
    path=np.array(path)
    t=np.linspace(0, 1, num=n, endpoint=True)
    rt1=path[:,0]
    rt2=path[:,1]       
    
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