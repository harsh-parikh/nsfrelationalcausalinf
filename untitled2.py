#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:18:45 2019

@author: harshparikh
"""
import numpy as np

def generate_table(n,p):
    idx = np.arange(0,n).reshape((n,1))
    cov = np.random.normal(0,1,size=(n,p))
    return np.hstack((idx,cov)) 

def generate_rel(t1,t2,rtype='manymany'):
    idx1 = t1[:,0]
    idx2 = t2[:,0]
    if rtype=='oneone':
        if len(idx1) == len(idx2):
            idx21 = np.random.permutations(idx2)
            return np.hstack((idx1,idx21)).reshape((len(idx1),2))
        else:
            return np.array([])
    elif rtype=='manymany':
        idxidx1 = np.random.randint(0,len(idx1),size=(len(idx1),))
        idxidx2 = np.random.randint(0,len(idx2),size=(len(idx1),))
        r = [ np.array([ idx1[ idxidx1[i] ], idx2[ idxidx2[i] ] ]) for i in range(0,len(idx1)) ]
        return np.array(r)
    elif rtype=='onemany':
        idxidx2 = np.random.randint(0,len(idx2),size=(len(idx1),))
        r = [ np.array([ idx1[i], idx2[ idxidx2[i] ] ]) for i in range(0,len(idx1)) ]
        return np.array(r)
    
    