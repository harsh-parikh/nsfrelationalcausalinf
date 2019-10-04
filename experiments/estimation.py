#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:28:41 2019

@author: harshparikh
"""

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.cm as cm

from data_gen import *

np.random.seed(0) #reproducability
df = generate_data(10000,200,75000,50)
df_inst = df['institutes']
df_auth = df['authors']
df_conf = df['conferences']
df_paper = df['papers']


fig = plt.figure(figsize=(8.5,9.5))
plt.scatter(df_paper['quality'],df_paper['review'],c=(df_conf.loc[df_paper['venue']]['single-blind']),alpha=0.4,cmap=cm.Dark2)
plt.colorbar()
plt.xlabel('Paper Quality')
plt.ylabel('Paper Review Score')
fig.savefig('Figures/quality_vs_review.png')

fig = plt.figure(figsize=(8.5,9.5))
sns.distplot(df_paper['review'],hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
fig.savefig('Figures/pdf_review.png')

# Mean as the embedding
fl = open('Logs/mean_estimate.csv','w')
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

print(',Mean Difference of Reviews (Single-Double), %f'%(review_diff),file=fl)
print('Single Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_single_ate),file=fl)
print('Double Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_double_ate),file=fl)

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_single_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_single_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Single-Blind')
fig.savefig('Figures/Mean/pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Mean/pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','mean_citation']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','mean_citation']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','mean_citation']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','mean_citation']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','mean_citation']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','mean_citation']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','mean_citation']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','mean_citation']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0


fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(tau_single,hist=False,kde_kws={'shade': True})
sns.distplot(tau_double,hist=False,kde_kws={'shade': True})
plt.axvline(ate_single,color='r',linestyle='--',alpha=0.6)
plt.axvline(mediante_single,color='g',linestyle='-',alpha=0.6)
plt.axvline(truth_single, color='b', linestyle='-',alpha=0.6)
plt.axvline(ate_double,color='y',linestyle='--',alpha=0.6)
plt.axvline(mediante_double,color='m',linestyle='-',alpha=0.6)
plt.axvline(truth_double, color='c', linestyle='-',alpha=0.6)
plt.legend([r'Mean $\tau$ Single-Blind',r'Median $\tau$ Single-Blind',r'True $\tau$ Single-Blind',r'Mean $\tau$ Double-Blind',r'Median $\tau$ Double-Blind',r'True $\tau$ Double-Blind','Single-Blind','Double-Blind'])
plt.xlabel('Estimated Treatment Effect')
plt.ylabel('Probability Density Estimate')
plt.title(r'Single Blind vs Double Blind $\tau$ s')
fig.savefig('Figures/Mean/pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)

fl.close()
#---------------------------------------------------------
# Median as the embedding
fl = open('Logs/median_estimate.csv','w')
df_unit_table = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    median_prestige = np.median(prestige_vec)
    median_citation = np.median(citation_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['median_prestige'] = median_prestige
    d_paperi['median_citation'] = median_citation
    df_unit_table[i] = d_paperi

df_unit_table = pd.DataFrame.from_dict(df_unit_table,orient='index')

df_single = df_unit_table.loc[df_unit_table['venue_single-blind'] == 1]
df_double = df_unit_table.loc[df_unit_table['venue_single-blind'] == 0]
df_single_treated = df_single.loc[df_single['median_prestige']>10]
df_single_control = df_single.loc[df_single['median_prestige']<=10]
df_double_treated = df_double.loc[df_double['median_prestige']>10]
df_double_control = df_double.loc[df_double['median_prestige']<=10]

review_diff = np.mean(df_single['review']) - np.mean(df_double['review'])
naive_single_ate = np.mean(df_single_treated['review']) - np.mean(df_single_control['review'])
naive_double_ate = np.mean(df_double_treated['review']) - np.mean(df_double_control['review'])

print(',Mean Difference of Reviews (Single-Double), %f'%(review_diff),file=fl)
print('Single Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_single_ate),file=fl)
print('Double Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_double_ate),file=fl)

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_single_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_single_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Single-Blind')
fig.savefig('Figures/Median/pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Median/pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','median_citation']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','median_citation']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','median_citation']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','median_citation']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','median_citation']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','median_citation']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','median_citation']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','median_citation']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0


fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(tau_single,hist=False,kde_kws={'shade': True})
sns.distplot(tau_double,hist=False,kde_kws={'shade': True})
plt.axvline(ate_single,color='r',linestyle='--',alpha=0.6)
plt.axvline(mediante_single,color='g',linestyle='-',alpha=0.6)
plt.axvline(truth_single, color='b', linestyle='-',alpha=0.6)
plt.axvline(ate_double,color='y',linestyle='--',alpha=0.6)
plt.axvline(mediante_double,color='m',linestyle='-',alpha=0.6)
plt.axvline(truth_double, color='c', linestyle='-',alpha=0.6)
plt.legend([r'Mean $\tau$ Single-Blind',r'Median $\tau$ Single-Blind',r'True $\tau$ Single-Blind',r'Mean $\tau$ Double-Blind',r'Median $\tau$ Double-Blind',r'True $\tau$ Double-Blind','Single-Blind','Double-Blind'])
plt.xlabel('Estimated Treatment Effect')
plt.ylabel('Probability Density Estimate')
plt.title(r'Single Blind vs Double Blind $\tau$ s')
fig.savefig('Figures/Median/pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)

fl.close()

#---------------------------------------------------------
#Complex Embedding

fl = open('Logs/complex_estimate.csv','w')
df_unit_table = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    median_prestige = np.percentile(prestige_vec,75)
    median_citation = scipy.special.expit(np.dot( [ 2**(-i) for i in range(len(citation_vec)) ], citation_vec )/500 - 1)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['quantile_75_prestige'] = median_prestige
    d_paperi['expit_citation'] = median_citation
    df_unit_table[i] = d_paperi

df_unit_table = pd.DataFrame.from_dict(df_unit_table,orient='index')

df_single = df_unit_table.loc[df_unit_table['venue_single-blind'] == 1]
df_double = df_unit_table.loc[df_unit_table['venue_single-blind'] == 0]
df_single_treated = df_single.loc[df_single['quantile_75_prestige']>10]
df_single_control = df_single.loc[df_single['quantile_75_prestige']<=10]
df_double_treated = df_double.loc[df_double['quantile_75_prestige']>10]
df_double_control = df_double.loc[df_double['quantile_75_prestige']<=10]

review_diff = np.mean(df_single['review']) - np.mean(df_double['review'])
naive_single_ate = np.mean(df_single_treated['review']) - np.mean(df_single_control['review'])
naive_double_ate = np.mean(df_double_treated['review']) - np.mean(df_double_control['review'])

print(',Mean Difference of Reviews (Single-Double), %f'%(review_diff),file=fl)
print('Single Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_single_ate),file=fl)
print('Double Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_double_ate),file=fl)

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_single_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_single_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Single-Blind')
fig.savefig('Figures/Complex_1/pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Complex_1/pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','expit_citation']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','expit_citation']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','expit_citation']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','expit_citation']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','expit_citation']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','expit_citation']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','expit_citation']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','expit_citation']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0


fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(tau_single,hist=False,kde_kws={'shade': True})
sns.distplot(tau_double,hist=False,kde_kws={'shade': True})
plt.axvline(ate_single,color='r',linestyle='--',alpha=0.6)
plt.axvline(mediante_single,color='g',linestyle='-',alpha=0.6)
plt.axvline(truth_single, color='b', linestyle='-',alpha=0.6)
plt.axvline(ate_double,color='y',linestyle='--',alpha=0.6)
plt.axvline(mediante_double,color='m',linestyle='-',alpha=0.6)
plt.axvline(truth_double, color='c', linestyle='-',alpha=0.6)
plt.legend([r'Mean $\tau$ Single-Blind',r'Median $\tau$ Single-Blind',r'True $\tau$ Single-Blind',r'Mean $\tau$ Double-Blind',r'Median $\tau$ Double-Blind',r'True $\tau$ Double-Blind','Single-Blind','Double-Blind'])
plt.xlabel('Estimated Treatment Effect')
plt.ylabel('Probability Density Estimate')
plt.title(r'Single Blind vs Double Blind $\tau$ s')
fig.savefig('Figures/Complex_1/pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)

fl.close()
