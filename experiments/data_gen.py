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
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

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
        blind = np.random.binomial(1,2/3)
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
        quality = scipy.special.expit(np.sum([ (2**(-i))*df_auth.loc[authors[i]]['citation'] for i in range(num_auth) ])/500 - 1)
        paper_conf = np.random.randint(0,len(df_conf)) #randomly apply to any conference, could be made better
        median_prestige = np.median([ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors]) #median prestige of all authors
        review_score = max(0,min(20,(df_conf.loc[paper_conf]['single-blind'])*(median_prestige>10) + (80*quality)/(df_conf.loc[paper_conf]['impact_factor']))) #if it is single-blind then the treatment effect of median-prestige=high is 1.
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
df = generate_data(1000000,100,10000,250)
df_inst = df['institutes']
df_auth = df['authors']
df_conf = df['conferences']
df_paper = df['papers']

fig = plt.figure(figsize=(8.5,9.5))
plt.scatter(df_paper['quality'],df_paper['review'],c=df_conf.loc[df_paper['venue']]['single-blind'])
plt.colorbar()
plt.xlabel('Paper Quality')
plt.ylabel('Paper Review Score')
fig.savefig('quality_vs_review.png')

fig = plt.figure(figsize=(8.5,9.5))
sns.distplot(df_paper['review'],hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3})
plt.xlim((0,20))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
fig.savefig('pdf_review.png')

df_unit_table = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    mean_prestige = np.mean(prestige_vec)
    mean_citation = np.mean(citation_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['mean_prestige'] = mean_prestige
    d_paperi['mean_citation'] = mean_citation
    df_unit_table[i] = d_paperi

df_unit_table = pd.DataFrame.from_dict(df_unit_table,orient='index')

df_single = df_unit_table.loc[df_unit_table['venue_single-blind'] == 1]
df_double = df_unit_table.loc[df_unit_table['venue_single-blind'] == 0]
df_single_treated = df_single.loc[df_single['mean_prestige']>10]
df_single_control = df_single.loc[df_single['mean_prestige']<=10]
df_double_treated = df_double.loc[df_double['mean_prestige']>10]
df_double_control = df_double.loc[df_double['mean_prestige']<=10]

review_diff = np.mean(df_single['review']) - np.mean(df_double['review'])
naive_single_ate = np.mean(df_single_treated['review']) - np.mean(df_single_control['review'])
naive_double_ate = np.mean(df_double_treated['review']) - np.mean(df_double_control['review'])

print(review_diff)
print(naive_single_ate)
print(naive_double_ate)

fig = plt.figure(figsize=(8.5,9.5))
sns.distplot(df_single_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_single_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,20))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
fig.savefig('pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(8.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,20))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
fig.savefig('pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','mean_citation']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','mean_citation']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','mean_citation']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','mean_citation']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

fig = plt.figure(figsize=(8.5,9.5))
sns.distplot(tau_single,hist=False,kde_kws={'shade': True})
plt.axvline(ate_single,color='r',linestyle='--',alpha=0.3)
plt.axvline(mediante_single,color='g',linestyle='-',alpha=0.3)
plt.axvline(truth_single, color='b', linestyle='-',alpha=0.3)
plt.legend({'ATE':ate_single,'Median':mediante_single,'Truth':truth_single})
plt.xlabel('Estimated Treatment Effect')
plt.ylabel('Probability Density Estimate')
plt.title('Single Blind')
fig.savefig('pdf_single_cate.png')

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','mean_citation']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','mean_citation']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','mean_citation']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','mean_citation']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

fig = plt.figure(figsize=(8.5,9.5))
sns.distplot(tau_double,hist=False,kde_kws={'shade': True})
plt.axvline(ate_double,color='r',linestyle='--',alpha=0.3)
plt.axvline(mediante_double,color='g',linestyle='-',alpha=0.3)
plt.axvline(truth_double, color='b', linestyle='-',alpha=0.3)
plt.legend({'ATE':ate_double,'Median':mediante_double,'Truth':truth_double})
plt.xlabel('Estimated Treatment Effect')
plt.ylabel('Probability Density Estimate')
plt.title('Double Blind')
fig.savefig('pdf_double_cate.png')