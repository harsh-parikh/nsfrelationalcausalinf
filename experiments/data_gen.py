#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:48:03 2019

@author: harshparikh
"""
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

def roll(bias_list):
    number = np.random.uniform(0, sum(bias_list))
    current = 0
    for i, bias in enumerate(bias_list):
        current += bias
        if number <= current:
            return i

## Generate Authors
def gen_inst(n_inst):
    def gen():
        inst_prestige = np.random.exponential(10)
        return [inst_prestige]
    d_inst = {}
    for i in range(0,n_inst):
        d_insti = {}
        inst = gen()
        d_insti['prestige'] = inst[0]
        d_inst[i] = d_insti
    d_inst = pd.DataFrame.from_dict(d_inst,orient='index')
    return d_inst

def gen_author(n_auth,df_inst):
    def gen():
        x = np.random.normal(30,10) 
        experience = x * (x > 0) #experience is normally distributed but always positive
        gender = np.random.binomial(1,1/2) #gender 0: male or 1: female
        w = [np.random.binomial(1,1/2), np.random.binomial(1,1/3), np.random.binomial(1,1/4), np.random.binomial(1,1/16)]
        x = [np.random.poisson(100), np.random.poisson(200), np.random.poisson(500), np.random.poisson(1000)]
        citation = int(np.dot(w,x)*experience/30) #citation count as generated using multimodal poisson distribution
        expertise = np.random.randint(0,10) #field of expertise
        high = np.array(df_inst['prestige']) #prestige of each school
        low = 10/high #inverse of prestige of school normalized by 10
        inv_cit = 304.166/(citation+1) #inverse of citation count times a normalizing factor
        affiliation_prob = scipy.special.expit(0.1*high*citation + 0.1*low*inv_cit - 0.1*high*inv_cit - 0.1*low*citation) #prob((a,i)) is \prop I[high]A[high] + I[low]A[low] - I[high]A[low] - I[low]A[high]
        affiliation_prob = affiliation_prob/sum(affiliation_prob) #normalizing
        auth_inst_id = roll(affiliation_prob) #rolling a biased dice
        return [gender,experience,citation,expertise,auth_inst_id]
    d_auth = {}
    for i in range(0,n_auth):
        d_authi = {}
        auth = gen()
        d_authi['age'] = auth[0]
        d_authi['experience'] = auth[1]
        d_authi['citation'] = auth[2]
        d_authi['expertise'] = auth[3]
        d_authi['affiliation'] = auth[4]
        d_auth[i] = d_authi
    return pd.DataFrame.from_dict(d_auth,orient='index')

def gen_conf(n_conf):
    def gen():
        impact_fac = np.random.exponential(10)
        area = np.random.randint(0,10)
        blind = np.random.binomial(1,1/3)
        return [area,impact_fac,blind]
    d_conf = {}
    for i in range(n_conf):
        d_confi = {}
        conf = gen()
        d_confi['area'] = conf[0]
        d_confi['impact_factor'] = conf[1]
        d_confi['single-blind'] = conf[2]
        d_conf[i] = d_confi
    return pd.DataFrame.from_dict(d_conf,orient='index')

def gen_paper(n_paper,df_conf,df_auth,df_inst):
    def gen():
        num_auth = int(np.random.exponential(2.5) + 1)
        authors = np.random.choice(len(df_auth), size = num_auth, replace=False) #randomly choose K authors, can be made more meaningful
        quality = scipy.special.expit(np.random.normal(0,1) + np.sum([ (2**(-i))*df_auth.loc[authors[i]]['citation'] for i in range(num_auth) ])/1000) - 0.5
        quality = quality * (quality > 0)
        paper_conf = np.random.randint(0,len(df_conf)) #randomly apply to any conference, could be made better
        median_prestige = np.median([ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors]) #median prestige of all authors
        review_score = (df_conf.loc[paper_conf]['single-blind'])*(median_prestige>10) + (10*quality)/(df_conf.loc[paper_conf]['impact_factor']) #if it is single-blind then the treatment effect of median-prestige=high is 1.
        return [authors,quality,paper_conf,review_score]
    d_paper = {}
    for i in range(0,n_paper):
        d_paperi = {}
        paper = gen()
        d_paperi['authors'] = paper[0]
        d_paperi['quality'] = paper[1]
        d_paperi['venue'] = paper[2]
        d_paperi['review'] = paper[3]
        d_paper[i] = d_paperi
    return pd.DataFrame.from_dict(d_paper,orient='index')
        
    
def generate_data(n_auth,n_inst,n_paper,n_conf):
    df_inst = gen_inst(n_inst)
    df_auth = gen_author(n_auth,df_inst)
    df_conf = gen_conf(n_conf)
    df_paper = gen_paper(n_paper,df_conf,df_auth,df_inst)
    df = {'institutes': df_inst, 'authors': df_auth, 'conferences': df_conf, 'papers': df_paper}
    return df

np.random.seed(0) #reproducability
df = generate_data(1000,100,10000,50)

