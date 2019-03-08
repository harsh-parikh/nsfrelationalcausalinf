#input subject-id-column (sid) , summarize-columns
import numpy as np
import scipy
from sklearn.neighbors.kde import KernelDensity

def summarize_set(S,mode='average'):
    # summarize the S and outputs a summary array
    # different modes, currently implemented average and normal
    # S = n x p form with p covariates and n samples
    n,p = S.shape
    if(mode=='average'):
        return np.average(S,axis=0)
    if(mode=='normal'):
        mu = np.average(S,axis=0)
        sigma = np.var(S,axis=0)
        outputarray = np.zeros((2*p,))
        for i in range(0,p):
            outputarray[2*i] = mu[i]
            outputarray[2*i+1] = sigma[i]
        return outputarray
    
def subset(D,colid,i,method='naive'):
    # given the subject-id =i of parent, find the subset S in D which has sid = i
    # the first column is unit id and while column id for sid is given as input 
    # i notes the value of sid
    S = []
    for row in D:
        if row[sid]==i:
            S.append(row)
    S

def summarize_child(Dparent,Dchild,sid):
    unique_sid = np.unique(Dparent[:,0])
    S = {}
    for i in unique_sid:
        s = summarize_set(subset(Dchild,sid,i))
        S[i] = s
        
    
def form_unit(T,root):
    children = get_children(T,root)
    if len(children)!=0:
        children_summary =  
            
    