#input subject-id-column (sid) , summarize-columns
import numpy as np
import scipy
from sklearn.neighbors.kde import KernelDensity

def set_summary(S,mode='average'):
    # summarize the S and outputs a summary array
    # different modes, currently implemented average and normal
    # S = n x p form with p covariates and n samples
    # ?How will the function know which columns are covariates and which are ids?
    # ?What if we want different method per covariate?
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
    return np.array(S)

def parent_child_aggregate(Dparent,Dchild,sid):
    S = {}
    Dparent1 = []
    for i in unique_sid:
        s = set_summary(subset(Dchild,sid,i))
        S[i] = s
    for row in Dparent:
        s = S[row[0]]
        snew = np.append(row,s)
        Dparent1.append(snew)
    return Dparent1

def loadfromfile(path):
    return 0

def savetofile(path):
    return 0
    
def summarizechildren():
    return 0

def form_unit(T,root):
    # method will recursive summarize the tree, bottom up
    # for each parent-child link it performs parent child aggregate
    # because it is bottom up, when a parent is aggregating information in child, the child is already aggregated for it's further children
    # tree is stored in a adjacency dictionary
    # dictionary has form {node: children-array, link-to-data}
    # children-array consists only of node-ids of child entities
    children = get_children(T,root)
    if len(children)!=0:
        children_summary = list(map(lambda x: form_unit(T,x),children))
    return 0
    