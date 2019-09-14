#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:12:51 2019

@author: harshparikh
"""
# relationship map
# D = { node : [psolid, [  (Rneighbor1,cid1,Sneighbor1), (Rneighbor2,cid2,Sneighbor2) ... , (RneighborN,cidN,SneighborN) ], link_to_table ] }


import numpy as np
import scipy 
import pandas as pd
import scipy.stats as stats

'''
Here is our system architecture:  
We accept as input a set of NSEs and a causal query. 
The input will be passed through the covariate detection module. 
This module identifies a set of sufficient covariates and embeddings  
required to answer the causal query at hand. Once the embeddings 
are computed using the relational embedding module we end up with 
a flat table consist of a treatment, covariate and outcome in Rubin 
format if you wish to see it that way. The table will be passed 
through the estimation module. The only place that the relational 
schema plays a role is in computing the embeddings.
'''

def readStructuralEqn(file):
    #reads the structural eqns from the given file
    return 0

def getCausalQuery(file):
    #reads the causal question from the given file
    return 0




#summarize_set takes in a set of vectors S and a list of p summary functions F. It applies each entry of F to corresponding dimension of S
def summarize_set( S, F ):
    n,p = S.shape
    summary_vec = np.zeros( ( p, ) )
    for k in range(0,p):
        summary_vec[k] = F[k]( np.array(S.iloc[:,k]) )
    return summary_vec

#finds the subset in set S using the relation table R such that value in column colid1 of R is equal to index
def find_subset( index, colid1, colid2, spcolid, S, R ):
    S1 = pd.DataFrame(columns=list(S.columns))
    for index, row in R.iterrows():
        if row[colid1] == index:
            index2 = row[colid2]
            S1 = S1.append( S.loc[index2] ) 
    return S1

#gets the table S from filesystem. S is the name of the table and D is the dictionary of tables.
def get(S,D,ftype='direct'):
    if ftype == 'direct':
        return D[S][2]
    else:
        path = D[S][2]
        df = pd.read_csv(path,sep=',',header=None)
        return df

#put a table noted by S in respective path mentioned in dictionary D
def put(S,STable,D,ftype = 'direct'):
    if ftype == 'direct':
        D[S][2] = pd.DataFrame(STable)
    else:
        df = pd.DataFrame(STable)
        path = D[S][2]
        df.to_csv(path, header=None, index=None)
    return 0

#returns primary identifier column id
def getpcolid(S,D):
    return D[S][0]

#returns neighbors of S who are connected by some relation.
def getneighbors(S,D):
    return D[S][1]

#summarize_neighbors functions takes in a root node and the dictionary D of the relations
def summarize_neighbors(S,D,visited):
    pcolid = getpcolid(S,D)
    neighbors = getneighbors(S,D)
    STable = get(S,D)
    summary_mat = pd.DataFrame()
    for rel in neighbors:
        (Rneighbor, colid1, colid2, Sneighbor) = rel
        print(Rneighbor, visited)
        if Rneighbor not in visited:
            visited = visited + [Rneighbor,Sneighbor]
            spcolid = getpcolid(Sneighbor,D)
            visited = summarize_neighbors(Sneighbor,D,visited)
            RneighborTable = get(Rneighbor,D)
            SneighborTable = get(Sneighbor,D)
            summary_mat_i = pd.DataFrame( columns = list(SneighborTable.columns) )
            for itr, row in STable.iterrows():
                index = row[pcolid]
                S1 = find_subset(index,colid1,colid2,spcolid,SneighborTable,RneighborTable)
                n1,p1 = S1.shape
                summary_vec = summarize_set( S1, [np.average]*p1  ) #can use multiple moments here
                summary_vec = pd.DataFrame([summary_vec],columns=summary_mat_i.columns,index=[index])
                summary_mat_i = summary_mat_i.append(summary_vec)
            summary_mat = pd.concat([summary_mat, summary_mat_i], axis=1)
#    summary_mat = np.hstack(tuple(summary_mat))
    STable = pd.concat( [ STable, summary_mat ], axis = 1 )
    put(S,STable,D)
    return visited

#wrapper that makes unit table
def make_unit(S,D):
    summarize_neighbors(S,D,[])
    return get(S,D)

#learns moments for set S where k is the number of moments we are interested in.
def learn_moment_summary(S,k=1):
    s = np.array([stats.moment(S,moment=i,axis=0) for i in range(0,k)])
    return s



