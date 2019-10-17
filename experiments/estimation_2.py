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
import scipy.stats as stats

from data_gen import *
from learn_embedding import *

np.random.seed(0) #reproducability
df = generate_data(1000,20,7500,50)
df_inst = df['institutes']
df_auth = df['authors']
df_conf = df['conferences']
df_paper = df['papers']


fig = plt.figure(figsize=(8.5,9.5))
plt.scatter(df_paper['quality'],df_paper['review'],alpha=0.4)
plt.xlabel('Paper Quality')
plt.ylabel('Paper Review Score')
fig.savefig('Figures/quality_vs_review.png')

fig = plt.figure(figsize=(8.5,9.5))
sns.distplot(df_paper['review'],hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
fig.savefig('Figures/pdf_review.png')

'''
df_tau_latent = {}
#---------------------------------------------------------
# Mean as the embedding
#---------------------------------------------------------
fl = open('Logs/Latent_Estimate.csv','w')
df_unit_table = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    mean_prestige = np.percentile(prestige_vec,75)
    mean_citation = np.mean(citation_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['mean_prestige'] = mean_prestige
    d_paperi['mean_citation'] = mean_citation
    d_paperi['embedded_experience'] = np.mean(experience_vec)
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

print('Mean as Embedding',file=fl)
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
fig.savefig('Figures/Mean/Latent_pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Mean/Latent_pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','mean_citation','embedded_experience']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','mean_citation','embedded_experience']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','mean_citation','embedded_experience']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','mean_citation','embedded_experience']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','mean_citation','embedded_experience']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','mean_citation','embedded_experience']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','mean_citation','embedded_experience']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','mean_citation','embedded_experience']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

df_tau_latent['mean_single'] = tau_single
df_tau_latent['mean_double'] = tau_double
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
fig.savefig('Figures/Mean/Latent_pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)

#---------------------------------------------------------
# Median as the embedding
#---------------------------------------------------------

df_unit_table = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    median_prestige = np.percentile(prestige_vec,75)
    median_citation = np.median(citation_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['median_prestige'] = median_prestige
    d_paperi['median_citation'] = median_citation
    d_paperi['embedded_experience'] = np.median(experience_vec)
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

print('Median as Embedding',file=fl)
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
fig.savefig('Figures/Median/Latent_pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Median/Latent_pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','median_citation','embedded_experience']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','median_citation','embedded_experience']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','median_citation','embedded_experience']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','median_citation','embedded_experience']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','median_citation','embedded_experience']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','median_citation','embedded_experience']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','median_citation','embedded_experience']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','median_citation','embedded_experience']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

df_tau_latent['median_single'] = tau_single
df_tau_latent['median_double'] = tau_double
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
fig.savefig('Figures/Median/Latent_pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)



#---------------------------------------------------------
#Complex Embedding
#---------------------------------------------------------

df_unit_table = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
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
    d_paperi['embedded_experience'] = np.max(experience_vec)
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
print('True Embedding',file=fl)
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
fig.savefig('Figures/Complex_1/Latent_pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Complex_1/Latent_pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','expit_citation','embedded_experience']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','expit_citation','embedded_experience']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','expit_citation','embedded_experience']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','expit_citation','embedded_experience']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','expit_citation','embedded_experience']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','expit_citation','embedded_experience']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','expit_citation','embedded_experience']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','expit_citation','embedded_experience']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

df_tau_latent['complex_single'] = tau_single
df_tau_latent['complex_double'] = tau_double
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
fig.savefig('Figures/Complex_1/Latent_pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)

##---------------------------------------------------------
##Learn RF + Moment Summary Embedding
##---------------------------------------------------------
#

df_unit_table_1 = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['prestige'] = prestige_vec
    d_paperi['citation'] = citation_vec
    d_paperi['experience'] = experience_vec
    df_unit_table_1[i] = d_paperi
    
df_unit_table_1 = pd.DataFrame.from_dict(df_unit_table_1,orient='index') 
embedding_needed_covariates = ['citation','experience']
embedding_cov = []
y = np.array(df_unit_table_1['review'])
for cov in embedding_needed_covariates:
    X_cov = df_unit_table_1[cov]
    output = learn_moment_summary(X_cov,y,learn_type='regression',max_moment=10)
    embedding_cov.append( lambda x: output[2].predict( np.array([[np.mean(x)]+[stats.moment(x,moment=i) for i in range(2,output[1])]]) )[0] )
    

df_unit_table = {}
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    embedded_prestige = np.percentile(prestige_vec,75)
    embedded_citation = embedding_cov[0](citation_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['embedded_prestige'] = embedded_prestige
    d_paperi['embedded_citation'] = embedded_citation
    d_paperi['embedded_experience'] = embedding_cov[1](experience_vec)
    df_unit_table[i] = d_paperi


df_unit_table = pd.DataFrame.from_dict(df_unit_table,orient='index')

df_single = df_unit_table.loc[df_unit_table['venue_single-blind'] == 1]
df_double = df_unit_table.loc[df_unit_table['venue_single-blind'] == 0]
df_single_treated = df_single.loc[df_single['embedded_prestige']>10]
df_single_control = df_single.loc[df_single['embedded_prestige']<=10]
df_double_treated = df_double.loc[df_double['embedded_prestige']>10]
df_double_control = df_double.loc[df_double['embedded_prestige']<=10]

review_diff = np.mean(df_single['review']) - np.mean(df_double['review'])
naive_single_ate = np.mean(df_single_treated['review']) - np.mean(df_single_control['review'])
naive_double_ate = np.mean(df_double_treated['review']) - np.mean(df_double_control['review'])

print('Learned Embedding using Moment Summarization followed by Random Forest',file=fl)
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
fig.savefig('Figures/Learn_MomSum_RF/Latent_pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Learn_MomSum_RF/Latent_pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','embedded_citation']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','embedded_citation']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','embedded_citation']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','embedded_citation']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','embedded_citation']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','embedded_citation']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','embedded_citation']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','embedded_citation']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

df_tau_latent['learn_comsum_rf_single'] = tau_single
df_tau_latent['learn_comsum_rf_double'] = tau_double
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
fig.savefig('Figures/Learn_MomSum_RF/Latent_pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)


#---------------------------------------------------------
#Learn Moment Summary Embedding
#---------------------------------------------------------


df_unit_table_1 = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['prestige'] = prestige_vec
    d_paperi['citation'] = citation_vec
    d_paperi['experience'] = experience_vec
    df_unit_table_1[i] = d_paperi
    
df_unit_table_1 = pd.DataFrame.from_dict(df_unit_table_1,orient='index') 
embedding_needed_covariates = ['citation','experience']
embedding_cov = []
y = np.array(df_unit_table_1['review'])
for cov in embedding_needed_covariates:
    X_cov = df_unit_table_1[cov]
    output = learn_moment_summary(X_cov,y,learn_type='regression',max_moment=10)
    embedding_cov.append( lambda x: [np.mean(x)]+[stats.moment(x,moment=i) for i in range(2,output[1])] ) 
    

df_unit_table = {}
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    embedded_prestige = np.percentile(prestige_vec,75)
    embedded_citation = embedding_cov[0](citation_vec)
    embedded_experience = embedding_cov[1](experience_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['embedded_prestige'] = embedded_prestige
    for k in range(0,len(embedded_citation)):
        d_paperi['embedded_citation_%d'%(k)] = embedded_citation[k]
    for k in range(0,len(embedded_experience)):
        d_paperi['embedded_experience_%d'%(k)] = embedded_experience[k]
    df_unit_table[i] = d_paperi


df_unit_table = pd.DataFrame.from_dict(df_unit_table,orient='index')

df_single = df_unit_table.loc[df_unit_table['venue_single-blind'] == 1]
df_double = df_unit_table.loc[df_unit_table['venue_single-blind'] == 0]
df_single_treated = df_single.loc[df_single['embedded_prestige']>10]
df_single_control = df_single.loc[df_single['embedded_prestige']<=10]
df_double_treated = df_double.loc[df_double['embedded_prestige']>10]
df_double_control = df_double.loc[df_double['embedded_prestige']<=10]

review_diff = np.mean(df_single['review']) - np.mean(df_double['review'])
naive_single_ate = np.mean(df_single_treated['review']) - np.mean(df_single_control['review'])
naive_double_ate = np.mean(df_double_treated['review']) - np.mean(df_double_control['review'])

print('Learned Embedding using Moment Summarization',file=fl)
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
fig.savefig('Figures/Learn_MomSum/Latent_pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Learn_MomSum/Latent_pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor()
m_s_d = RandomForestRegressor()

m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor']+['embedded_citation_'+str(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]]) - m_s_c.predict(df_double[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor()
m_d_d = RandomForestRegressor()

m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))] ],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))] ],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]]) - m_d_c.predict(df_single[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

df_tau_latent['learn_comsum_single'] = tau_single
df_tau_latent['learn_comsum_double'] = tau_double
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
fig.savefig('Figures/Learn_MomSum/Latent_pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)



fig = plt.figure(figsize=(10.5,9.5))

plt.axhline(y=1,color='m',linestyle='--',alpha=0.3)
plt.violinplot(df_tau_latent['mean_single'],positions=[1],showmeans=True)
plt.violinplot(df_tau_latent['median_single'],positions=[2],showmeans=True)
plt.violinplot(df_tau_latent['complex_single'],positions=[3],showmeans=True)
plt.violinplot(df_tau_latent['learn_comsum_rf_single'],positions=[4],showmeans=True)
plt.violinplot(df_tau_latent['learn_comsum_single'],positions=[5],showmeans=True)

plt.axhline(y=0,color='c',linestyle='--',alpha=0.3)
plt.violinplot(df_tau_latent['mean_double'],positions=[6],showmeans=True)
plt.violinplot(df_tau_latent['median_double'],positions=[7],showmeans=True)
plt.violinplot(df_tau_latent['complex_double'],positions=[8],showmeans=True)
plt.violinplot(df_tau_latent['learn_comsum_rf_double'],positions=[9],showmeans=True)
plt.violinplot(df_tau_latent['learn_comsum_double'],positions=[10],showmeans=True)

plt.xlabel('CATEs')
plt.xticks(list(np.arange(1,11)),['mean_single','median_single','complex_single','learn_comsum_rf_single','learn_comsum_single','mean_double','median_double','complex_double','learn_comsum_rf_double','learn_comsum_double'], rotation=75)
plt.legend(['True TE Single-Blind','True TE Double-Blind'])
plt.tight_layout()
fig.savefig('Figures/Latent_violin_single_double_cate.png')
fl.close()
'''

#---------------------------------------------------------
#---------------------------------------------------------
#Removing the Latent Variable for ``Quality"
#---------------------------------------------------------
#---------------------------------------------------------


df_tau = {}

#---------------------------------------------------------
# Mean as the embedding
#---------------------------------------------------------
fl = open('Logs/Estimate.csv','w')
df_unit_table = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    mean_prestige = np.percentile(prestige_vec,75)
    mean_citation = np.mean(citation_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['mean_prestige'] = mean_prestige
    d_paperi['mean_citation'] = mean_citation
    d_paperi['embedded_experience'] = np.mean(experience_vec)
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

print('Mean as Embedding',file=fl)
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

m_s_c = RandomForestRegressor(n_estimators=1000)
m_s_d = RandomForestRegressor(n_estimators=1000)

m_s_c = m_s_c.fit(df_single_control[['venue_impact_factor','mean_citation','embedded_experience']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['venue_impact_factor','mean_citation','embedded_experience']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['venue_impact_factor','mean_citation','embedded_experience']]) - m_s_c.predict(df_double[['venue_impact_factor','mean_citation','embedded_experience']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor(n_estimators=1000)
m_d_d = RandomForestRegressor(n_estimators=1000)

m_d_c = m_d_c.fit(df_double_control[['venue_impact_factor','mean_citation','embedded_experience']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['venue_impact_factor','mean_citation','embedded_experience']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['venue_impact_factor','mean_citation','embedded_experience']]) - m_d_c.predict(df_single[['venue_impact_factor','mean_citation','embedded_experience']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

df_tau['mean_single'] = tau_single
df_tau['mean_double'] = tau_double
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


#---------------------------------------------------------
# Median as the embedding
#---------------------------------------------------------

df_unit_table = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    median_prestige = np.percentile(prestige_vec,75)
    median_citation = np.median(citation_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['median_prestige'] = median_prestige
    d_paperi['median_citation'] = median_citation
    d_paperi['embedded_experience'] = np.median(experience_vec)
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

print('Median as Embedding',file=fl)
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

m_s_c = RandomForestRegressor(n_estimators=1000)
m_s_d = RandomForestRegressor(n_estimators=1000)

m_s_c = m_s_c.fit(df_single_control[['venue_impact_factor','median_citation','embedded_experience']],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['venue_impact_factor','median_citation','embedded_experience']],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['venue_impact_factor','median_citation','embedded_experience']]) - m_s_c.predict(df_double[['venue_impact_factor','median_citation','embedded_experience']])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor(n_estimators=1000)
m_d_d = RandomForestRegressor(n_estimators=1000)

m_d_c = m_d_c.fit(df_double_control[['venue_impact_factor','median_citation','embedded_experience']],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['venue_impact_factor','median_citation','embedded_experience']],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['venue_impact_factor','median_citation','embedded_experience']]) - m_d_c.predict(df_single[['venue_impact_factor','median_citation','embedded_experience']])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

df_tau['median_single'] = tau_single
df_tau['median_double'] = tau_double
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

#---------------------------------------------------------
#Learn Moment Summary Embedding
#---------------------------------------------------------


df_unit_table_1 = {}
n_paper = len(df_paper)
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['prestige'] = prestige_vec
    d_paperi['citation'] = citation_vec
    d_paperi['experience'] = experience_vec
    df_unit_table_1[i] = d_paperi
    
df_unit_table_1 = pd.DataFrame.from_dict(df_unit_table_1,orient='index') 
embedding_needed_covariates = ['citation','experience']
embedding_cov = []
y = np.array(df_unit_table_1['review'])
for cov in embedding_needed_covariates:
    X_cov = df_unit_table_1[cov]
    output = learn_moment_summary(X_cov,y,learn_type='regression',max_moment=7)
    embedding_cov.append( lambda x: moment_summarization([x],level=output[1])[0,:] ) 
    

df_unit_table = {}
for i in range(n_paper):
    paper =  df_paper.loc[i]
    authors = paper['authors']
    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
    experience_vec = [ df_auth.loc[a]['experience'] for a in authors ]
    embedded_prestige = np.percentile(prestige_vec,75)
    embedded_citation = embedding_cov[0](citation_vec)
    embedded_experience = embedding_cov[1](experience_vec)
    d_paperi = {}
    d_paperi['quality'] = paper['quality']
    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
    d_paperi['review'] = paper['review']
    d_paperi['embedded_prestige'] = embedded_prestige
    for k in range(0,len(embedded_citation)):
        d_paperi['embedded_citation_%d'%(k)] = embedded_citation[k]
    for k in range(0,len(embedded_experience)):
        d_paperi['embedded_experience_%d'%(k)] = embedded_experience[k]
    df_unit_table[i] = d_paperi


df_unit_table = pd.DataFrame.from_dict(df_unit_table,orient='index')

df_single = df_unit_table.loc[df_unit_table['venue_single-blind'] == 1]
df_double = df_unit_table.loc[df_unit_table['venue_single-blind'] == 0]
df_single_treated = df_single.loc[df_single['embedded_prestige']>10]
df_single_control = df_single.loc[df_single['embedded_prestige']<=10]
df_double_treated = df_double.loc[df_double['embedded_prestige']>10]
df_double_control = df_double.loc[df_double['embedded_prestige']<=10]

review_diff = np.mean(df_single['review']) - np.mean(df_double['review'])
naive_single_ate = np.mean(df_single_treated['review']) - np.mean(df_single_control['review'])
naive_double_ate = np.mean(df_double_treated['review']) - np.mean(df_double_control['review'])

print('Learned Embedding using Moment Summarization',file=fl)
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
fig.savefig('Figures/Learn_MomSum/pdf_single_treated_control_review.png')

fig = plt.figure(figsize=(10.5,9.5))
sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
plt.xlim((0,10))
plt.xlabel('Review Score')
plt.ylabel('Probability Density Estimate')
plt.legend(['control','treated'])
plt.title('Double-Blind')
fig.savefig('Figures/Learn_MomSum/pdf_double_treated_control_review.png')

m_s_c = RandomForestRegressor(n_estimators=1000)
m_s_d = RandomForestRegressor(n_estimators=1000)

m_s_c = m_s_c.fit(df_single_control[['venue_impact_factor']+['embedded_citation_'+str(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]],df_single_control['review'])
m_s_d = m_s_d.fit(df_single_treated[['venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]],df_single_treated['review'])
tau_single = m_s_d.predict(df_double[['venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]]) - m_s_c.predict(df_double[['venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]])
ate_single = np.mean(tau_single)
mediante_single = np.median(tau_single)
truth_single = 1.0

m_d_c = RandomForestRegressor(n_estimators=1000)
m_d_d = RandomForestRegressor(n_estimators=1000)

m_d_c = m_d_c.fit(df_double_control[['venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))] ],df_double_control['review'])
m_d_d = m_d_d.fit(df_double_treated[['venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))] ],df_double_treated['review'])
tau_double = m_d_d.predict(df_single[['venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]]) - m_d_c.predict(df_single[['venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]+['embedded_experience_'+str(k) for k in range(0,len(embedded_experience))]])
ate_double = np.mean(tau_double)
mediante_double = np.median(tau_double)
truth_double= 0.0

df_tau['learn_comsum_single'] = tau_single
df_tau['learn_comsum_double'] = tau_double
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
fig.savefig('Figures/Learn_MomSum/pdf_single_double_cate.png')
print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)



fig = plt.figure(figsize=(10.5,9.5))
plt.rcParams.update({'font.size': 16})
plt.axhline(y=1,color='m',linestyle='--',alpha=0.3)
plt.violinplot(df_tau['mean_single'],positions=[1],showmeans=True,showextrema=True)
plt.violinplot(df_tau['median_single'],positions=[2],showmeans=True,showextrema=True)
plt.violinplot(df_tau['complex_single'],positions=[3],showmeans=True,showextrema=True)
plt.violinplot(df_tau['learn_comsum_rf_single'],positions=[4],showmeans=True,showextrema=True)
plt.violinplot(df_tau['learn_comsum_single'],positions=[5],showmeans=True,showextrema=True)

#plt.ylim((-0.5,2))
plt.xlabel('CATEs')
plt.xticks(list(np.arange(1,6)),['Mean','Median','Complex','Learned: Moments+RF','Learned: Moments'], rotation=75)
plt.legend(['True TE Single-Blind'])
plt.tight_layout()
fig.savefig('Figures/violin_single_cate.png')

fig = plt.figure(figsize=(10.5,9.5))
plt.rcParams.update({'font.size': 16})
plt.axhline(y=0,color='m',linestyle='--',alpha=0.3)
plt.violinplot(df_tau['mean_double'],positions=[1],showmeans=True,showextrema=True)
plt.violinplot(df_tau['median_double'],positions=[2],showmeans=True,showextrema=True)
plt.violinplot(df_tau['complex_double'],positions=[3],showmeans=True,showextrema=True)
plt.violinplot(df_tau['learn_comsum_rf_double'],positions=[4],showmeans=True,showextrema=True)
plt.violinplot(df_tau['learn_comsum_double'],positions=[5],showmeans=True,showextrema=True)

#plt.ylim((-2,1))
plt.xlabel('CATEs')
plt.xticks(list(np.arange(1,6)),['Mean','Median','Complex','Learned: Moments+RF','Learned: Moments'], rotation=75)
plt.legend(['True TE Double-Blind'])
plt.tight_layout()
fig.savefig('Figures/violin_double_cate.png')

fl.close()