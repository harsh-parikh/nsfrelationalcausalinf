#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:12:51 2019

@author: harshparikh
"""
# relationship map
# D = { node : (psolid, [  (Rneighbor1,cid1,Sneighbor1), (Rneighbor2,cid2,Sneighbor2) ... , (RneighborN,cidN,SneighborN) ], link_to_table ) }


import numpy as np
import scipy 
import pandas as pd

def summarize_set( S, F ):
    n,p = S.shape
    summary_vec = np.zeros( ( p, ) )
    for k in range(0,p):
        summary_vec[k] = F[k]( S[:,k] )
    return summary_vec

def find_subset( index, colid1, colid2, spcolid, S, R ):
    S1 = []
    for row in R:
        if row[colid1] == index:
            index2 = row[colid2]
            S1 = S1 + [ S[index2,:] ]
    return S1

def get(S,D):
    path = D[S][2]
    df = pd.read_csv(path,sep=',',header=None)
    return df.to_numpy()

def put(S,STable,D):
    df = pd.DataFrame(STable)
    path = D[S][2]
    df.to_csv(path, header=None, index=None)
    return 0

def getpcolid(S,D):
    return D[S][0]

def getneighbors(S,D):
    return D[S][1]

def summarize_neighbors(S,D,visited):
    pcolid = getpcolid(S,D)
    neighbors = getneighbors(S,D)
    STable = get(S,D)
    summary_mat = []
    for rel in neighbors:
        summary_mat_i = []
        (Rneighbor,cid_neighbor,Sneighbor) = rel
        if Rneighbor not in visited:
            visited = visited + [Rneighbor,Sneighbor]
            spcolid = getpcolid(Sneighbor,D)
            summarize_neighbors(Sneighbor,spcolid,D,visited)
            RneighborTable = get(Rneighbor,D)
            SneighborTable = get(Sneighbor,D)
            for row in STable:
                index = row[pcolid]
                S1 = find_subset(index,1-cid_neighbor,cid_neighbor,spcolid,SneighborTable,RneighborTable)
                n1,p1 = S1.shape
                summary_vec = summarize_set( S1, [np.average]*p1  )
                summary_mat_i = summary_mat_i + [ summary_vec ]
            summary_mat = summary_mat + [ summary_mat_i ]
    summary_mat = np.hstack(tuple(summary_mat))
    STable = np.hstack( ( STable, summary_mat ) )
    put(S,STable,D)
    return 0

