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
# Average Treatment Effect
#---------------------------------------------------------

df_single_1 = df_single_unit_table.loc[df_single_unit_table['embedded_prestige'] >= 0.4]
df_single_0 = df_single_unit_table.loc[df_single_unit_table['embedded_prestige'] < 0.4]

df_double_1 = df_double_unit_table.loc[df_double_unit_table['embedded_prestige'] >= 0.4]
df_double_0 = df_double_unit_table.loc[df_double_unit_table['embedded_prestige'] < 0.4]

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
df_ate = pd.DataFrame()
cc_s = np.corrcoef(df_single_unit_table['embedded_prestige'],df_single_unit_table['decision'])[0,1]
cc_d = np.corrcoef(df_double_unit_table['embedded_prestige'],df_double_unit_table['decision'])[0,1]

fig = plt.figure(figsize=(15,15))
plt.rcParams.update({'font.size': 35})

df_ate = pd.DataFrame()
df_ate['Quantity'] = ['ATE','ATE','Correlation','Correlation']
df_ate['Estimates'] = [iso_ate_s,iso_ate_d,cc_s,cc_d]
df_ate['Venue'] = ['Single-Blind','Double-Blind','Single-Blind','Double-Blind']
splot = sns.barplot(x='Quantity',y='Estimates',hue='Venue',data=df_ate, palette="RdBu")
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.4f'), (p.get_x() + p.get_width() / 2., p.get_height()/2), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

#sns.distplot(tau_single,hist=False,kde_kws={'bw':0.4,'shade': True})
#sns.distplot(tau_double,hist=False,kde_kws={'bw':0.4,'shade': True})

plt.axhline(y=0,color='black',alpha=0.3)
#plt.axvline(cc_d,color='c')
#plt.xlim((-1,1))
#plt.xlabel('estimated CATEs')
#plt.ylabel('estimated probability density')
#plt.yscale('log')
plt.title('Review Data \nAverage Treatment Effects')
#plt.legend(['ATE, Single-Blind = %0.4f'%(iso_ate_s),'ATE, Double-Blind = %0.4f'%(iso_ate_d),'Single-Blind, CorrCoef = %0.4f'%(cc_s),'Double-Blind, CorrCoef = %0.4f'%(cc_d),'PDF TE Single-Blind','PDF TE Double-Blind'],loc='upper center',bbox_to_anchor=(0.5, -0.1),ncol=3)
#plt.tight_layout()
fig.savefig('Figures/real_ate.png')

corrm_s = df_single_unit_table.corr()
print(corrm_s.to_string())

corrm_d = df_double_unit_table.corr()
print(corrm_d.to_string())

#---------------------------------------------------------
# Isolated and Relational Effects
#---------------------------------------------------------

def read_df_auth():
    df_s_a = pd.read_csv('single_authors.csv')
    return df_s_a

df_s_a = read_df_auth()

df_s_a_emb = {}
for index, row in df_s_a.iterrows():
    author = row
    prestige = int(row['ego prestige'])
    citation = int(row['ego citations'])
    prestige_ca_vec = parse_as_list( row['relational prestige'] ,dtype='int')
    citation_ca_vec = parse_as_list(row['relational citations'],dtype='int')
    review = float(row['avg review score'])
    dpaperi = {}
    dpaperi['prestige'] = int(np.nan_to_num( np.exp(50) / ( np.exp(50) + np.exp(prestige) ) ) >= 0.35)
    dpaperi['citation'] = np.nan_to_num(np.mean(citation))
    dpaperi['collab_prestige'] = int(np.nan_to_num(np.median(np.exp(50) / ( np.exp(50) + np.exp(prestige_ca_vec) ))) >= 0.35)
    dpaperi['collab_citation'] = np.nan_to_num(np.mean(citation_ca_vec))
    dpaperi['review'] = scipy.special.expit(20*(review-0.5))
    df_s_a_emb[index] = dpaperi

dsa = pd.DataFrame.from_dict(df_s_a_emb,orient='index')

dsa11 = dsa.loc[dsa['prestige']==1].loc[dsa['collab_prestige']==1]
dsa10 = dsa.loc[dsa['prestige']==1].loc[dsa['collab_prestige']==0]
dsa01 = dsa.loc[dsa['prestige']==0].loc[dsa['collab_prestige']==1]
dsa00 = dsa.loc[dsa['prestige']==0].loc[dsa['collab_prestige']==0]

rfs11 = RFR(n_estimators = 50)
rfs10 = RFR(n_estimators = 50)
rfs01 = RFR(n_estimators = 50)
rfs00 = RFR(n_estimators = 50)

rfs11 = rfs11.fit(dsa11[['citation','collab_citation']],dsa11['review'])
rfs10 = rfs10.fit(dsa10[['citation','collab_citation']],dsa10['review'])
rfs01 = rfs01.fit(dsa01[['citation','collab_citation']],dsa01['review'])
rfs00 = rfs00.fit(dsa00[['citation','collab_citation']],dsa00['review'])

ie1 = rfs11.predict(dsa[['citation','collab_citation']]) - rfs01.predict(dsa[['citation','collab_citation']])
ie0 = rfs10.predict(dsa[['citation','collab_citation']]) - rfs00.predict(dsa[['citation','collab_citation']])

re1 = rfs11.predict(dsa[['citation','collab_citation']]) - rfs10.predict(dsa[['citation','collab_citation']])
re0 = rfs01.predict(dsa[['citation','collab_citation']]) - rfs00.predict(dsa[['citation','collab_citation']])

oe = rfs11.predict(dsa[['citation','collab_citation']]) - rfs00.predict(dsa[['citation','collab_citation']])

fig = plt.figure(figsize=(15,15))
plt.rcParams.update({'font.size': 35})

#plt.boxplot(df_tau['join_single'],positions=[1],showmeans=True,showfliers=False)
#plt.boxplot(list(ie0)+list(ie1),positions=[1],showmeans=True,showfliers=False)
#plt.boxplot(list(re0)+list(re1),positions=[2],showmeans=True,showfliers=False)
#plt.boxplot(oe,positions=[3],showmeans=True,showfliers=False)
#plt.xticks(list(np.arange(0,4)),['','Isolated','Relational','Overall'])
#plt.ylabel('Estimated Causal Effects')

#sns.distplot(list(ie0)+list(ie1))
#sns.distplot(list(re0)+list(re1))
#sns.distplot(oe)

df_ce = pd.DataFrame()
df_ce['Quantity'] = ['Correlation','AIE','ARE','AOE']
df_ce['Estimates'] = [np.corrcoef(dsa['prestige'],dsa['review'])[0,1],np.mean(list(ie0)+list(ie1)),np.mean(list(re0)+list(re1)),np.mean(oe)]
splot = sns.barplot(x='Quantity',y='Estimates',data=df_ce,palette='vlag' )
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.4f'), (p.get_x() + p.get_width() / 2., p.get_height()/2), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title(r'(ReviewData) Single-Blind')
fig.savefig('Figures/box_real_a.png')
