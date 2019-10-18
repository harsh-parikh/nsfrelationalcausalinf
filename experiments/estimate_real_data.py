#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:45:11 2019

@author: harshparikh
"""

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.cm as cm
import scipy.stats as stats
from learn_embedding import *

def read_df():
    df_single = pd.read_csv('single_df.csv')
    df_double = pd.read_csv('double_df.csv')
    return df_single, df_double

def parse_as_list(s,dtype='int'):
    if s=='[]':
        return []
    s1 = s.replace('[','').replace(']','').split(',')
    if len(s1)==0:
        return []
    else:
        if dtype == 'int':
            return list(map(int,s1))
        if dtype == 'bool':
            return list(map(lambda x: x=='True',s1))
        if dtype == 'float':
            return list(map(float,s1))

df_single, df_double = read_df()

np.random.seed(0)
#---------------------------------------------------------
# Mean as the embedding
#---------------------------------------------------------

df_single_emb = {}
for index, row in df_single.iterrows():
    paper = row
    prestige_vec = parse_as_list(row['isolated prestige'],dtype='int')
    citation_vec = parse_as_list(row['isolated citations'],dtype='int')
    prestige_ca_vec = parse_as_list( row['relational prestige'] ,dtype='int')
    citation_ca_vec = parse_as_list(row['relational citations'],dtype='int')
    review = row['review score']
    decision = row['decision']
    dpaperi = {}
    dpaperi['embedded_prestige'] = np.nan_to_num( np.median( np.exp(50) / ( np.exp(50) + np.exp(prestige_vec) ) ) )
    dpaperi['embedded_citation'] = np.nan_to_num(np.mean(citation_vec))
    dpaperi['embedded_collab_prestige'] = np.nan_to_num(np.median(np.exp(50) / ( np.exp(50) + np.exp(prestige_ca_vec) )))
    dpaperi['embedded_collab_citation'] = np.nan_to_num(np.mean(citation_ca_vec))
    dpaperi['review'] = np.mean(review)
    dpaperi['decision'] = decision
    df_single_emb[index] = dpaperi

df_single_unit_table = pd.DataFrame.from_dict(df_single_emb,orient='index')

df_double_emb = {}
for index, row in df_double.iterrows():
    paper = row
    prestige_vec = parse_as_list(row['isolated prestige'],dtype='int')
    citation_vec = parse_as_list(row['isolated citations'],dtype='int')
    prestige_ca_vec = parse_as_list( row['relational prestige'] ,dtype='int')
    citation_ca_vec = parse_as_list(row['relational citations'],dtype='int')
    review = row['review score']
    decision = row['decision']
    dpaperi = {}
    dpaperi['embedded_prestige'] = np.nan_to_num( np.median( np.exp(50) / ( np.exp(50) + np.exp(prestige_vec) ) ) )
    dpaperi['embedded_citation'] = np.nan_to_num(np.mean(citation_vec))
    dpaperi['embedded_collab_prestige'] = np.nan_to_num(np.median(np.exp(50) / ( np.exp(50) + np.exp(prestige_ca_vec) )))
    dpaperi['embedded_collab_citation'] = np.nan_to_num(np.mean(citation_ca_vec))
    dpaperi['review'] = review
    dpaperi['decision'] = decision
    df_double_emb[index] = dpaperi

df_double_unit_table = pd.DataFrame.from_dict(df_double_emb,orient='index')

#---------------------------------------------------------
# Isolated Effect
#---------------------------------------------------------

df_single_1 = df_single_unit_table.loc[df_single_unit_table['embedded_prestige'] >= 0.35]
df_single_0 = df_single_unit_table.loc[df_single_unit_table['embedded_prestige'] < 0.35]

df_double_1 = df_double_unit_table.loc[df_double_unit_table['embedded_prestige'] >= 0.35]
df_double_0 = df_double_unit_table.loc[df_double_unit_table['embedded_prestige'] < 0.35]

rf_single_1 = RFC(n_estimators = 100)
rf_single_0 = RFC(n_estimators = 100)

rf_double_1 = RFC(n_estimators = 100)
rf_double_0 = RFC(n_estimators = 100)

rf_single_1 = rf_single_1.fit(df_single_1[['embedded_citation','embedded_collab_prestige','embedded_collab_citation']],df_single_1['decision'])
rf_single_0 = rf_single_0.fit(df_single_0[['embedded_citation','embedded_collab_prestige','embedded_collab_citation']],df_single_0['decision'])

rf_double_1 = rf_double_1.fit(df_double_1[['embedded_citation','embedded_collab_prestige','embedded_collab_citation']],df_double_1['decision'])
rf_double_0 = rf_double_0.fit(df_double_0[['embedded_citation','embedded_collab_prestige','embedded_collab_citation']],df_double_0['decision'])

tau_single = rf_single_1.predict(df_single_unit_table[['embedded_citation','embedded_collab_prestige','embedded_collab_citation']]) - rf_single_0.predict(df_single_unit_table[['embedded_citation','embedded_collab_prestige','embedded_collab_citation']])
tau_double = rf_double_1.predict(df_double_unit_table[['embedded_citation','embedded_collab_prestige','embedded_collab_citation']]) - rf_double_0.predict(df_double_unit_table[['embedded_citation','embedded_collab_prestige','embedded_collab_citation']])

iso_ate_s = np.mean(tau_single)
iso_ate_d = np.mean(tau_double)

print(iso_ate_s)
print(iso_ate_d)

cc_s = np.corrcoef(df_single_unit_table['embedded_prestige'],df_single_unit_table['decision'])[0,1]
cc_d = np.corrcoef(df_double_unit_table['embedded_prestige'],df_double_unit_table['decision'])[0,1]
fig = plt.figure(figsize=(10.5,9.5))
plt.rcParams.update({'font.size': 16})
sns.distplot(tau_single,hist=False,kde_kws={'bw':0.4,'shade': True})
sns.distplot(tau_double,hist=False,kde_kws={'bw':0.4,'shade': True})
plt.axvline(np.mean(tau_single),color='b')
plt.axvline(np.mean(tau_double),color='r')
plt.axvline(cc_s,color='y')
plt.axvline(cc_d,color='c')
plt.xlim((-1,1))
plt.xlabel('estimated CATEs')
plt.ylabel('estimated probability density')
plt.title('PDF of Isolated TE (Embedding: Mean)')
plt.legend(['ATE, Single-Blind = %0.4f'%(iso_ate_s),'ATE, Double-Blind = %0.4f'%(iso_ate_d),'Single-Blind, CorrCoef = %0.4f'%(cc_s),'Double-Blind, CorrCoef = %0.4f'%(cc_d),'PDF TE Single-Blind','PDF TE Double-Blind'],loc='upper center',bbox_to_anchor=(0.5, -0.1),ncol=3)
plt.tight_layout()
fig.savefig('Figures/pdf_real_mean_single_double_cate.png')

corrm_s = df_single_unit_table.corr()
print(corrm_s.to_string())

corrm_d = df_double_unit_table.corr()
print(corrm_d.to_string())

#---------------------------------------------------------
# Learning embedding
#---------------------------------------------------------

embedding_needed_covariates = ['isolated citations','relational citations','relational prestige']
embedding_cov = []
y = np.array(df_single['review'])
for cov in embedding_needed_covariates:
    X_cov = df_single[cov]
    output = learn_moment_summary(X_cov,y,learn_type='regression',max_moment=7)
    embedding_cov.append( lambda x: moment_summarization([x],level=output[1])[0,:] ) 
