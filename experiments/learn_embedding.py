#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:37:00 2019

@author: harshparikh
"""
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_validate
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from data_gen import *

def eval_outcome_estimation(X,y,learn_type='regression'):
    if learn_type=='regression':
        try:
            model = RFR(n_estimators=500)
            scores = cross_validate(model,X,y,cv=5, return_estimator=True)
            return np.mean(scores['test_score']), scores['estimator'][np.argmax(scores['test_score'])]
        except:
            return 0, None
    else:
        try:
            model = RFC(n_estimators=500)
            scores = cross_validate(model,X,y,cv=5, return_estimator=True)
            return np.mean(scores['test_score']), scores['estimator'][np.argmax(scores['test_score'])]
        except:
            return 0, None
        
def moment_summarization(X,level=1):
    print(X)
    log_check = np.array([ (np.array(row)>0).all() for row in X ]).all()
    x = []
    for row in X:
        a = []
        if log_check:
            log_row = np.log(row)
            a += [np.mean(log_row),stats.gmean(row),stats.hmean(row)]
        a += [np.mean(row)]
        if level>1:
            a += [stats.moment(row,moment=i) for i in range(2,level)]
        x.append(a)
    return np.array(x)

def learn_moment_summary(X,y,learn_type='regression',log=False,max_moment=10):
    scores = np.nan_to_num(np.array([eval_outcome_estimation(moment_summarization(X,level=i),y,learn_type)[0] for i in range(1,max_moment)]))
    best_moment = np.argmax(scores)+1
    return scores, best_moment, eval_outcome_estimation(moment_summarization(X,level=best_moment),y,learn_type)[1]
#
#X = np.array([[1,2,3,4,5],[2,3],[10,10,12],[1,2,34,10,10,1,2,34,1],[1,2,3,4,5],[2,3],[10,10,12],[1,2,34,10,10,1,2,34,1],[1,2,3,4,5],[2,3],[10,10,12],[1,2,34,10,10,1,2,34,1],[1,2,3,4,5],[2,3],[10,10,12],[1,2,34,10,10,1,2,34,1],[1,2,3,4,5],[2,3],[10,10,12],[1,2,34,10,10,1,2,34,1]])
#y = [np.median(X[i]) for i in range(0,len(X))]
#output = learn_moment_summary(X,y,learn_type='regression',max_moment=5)


#---------------------------------------------------------
# Data Generation
#---------------------------------------------------------
#np.random.seed(0) #reproducability
#df = generate_data(1000,20,10000,50)
#df_inst = df['institutes']
#df_auth = df['authors']
#df_conf = df['conferences']
#df_paper = df['papers']
#fl = open('Logs/learned_embedding_momsum_estimate.csv','w')
###---------------------------------------------------------
###Learn RF + Moment Summary Embedding
###---------------------------------------------------------
##
#
#df_unit_table_1 = {}
#n_paper = len(df_paper)
#for i in range(n_paper):
#    paper =  df_paper.loc[i]
#    authors = paper['authors']
#    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
#    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
#    d_paperi = {}
#    d_paperi['quality'] = paper['quality']
#    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
#    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
#    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
#    d_paperi['review'] = paper['review']
#    d_paperi['prestige'] = prestige_vec
#    d_paperi['citation'] = citation_vec
#    df_unit_table_1[i] = d_paperi
#    
#df_unit_table_1 = pd.DataFrame.from_dict(df_unit_table_1,orient='index') 
#embedding_needed_covariates = ['citation']
#embedding_cov = []
#y = np.array(df_unit_table_1['review'])
#for cov in embedding_needed_covariates:
#    X_cov = df_unit_table_1[cov]
#    output = learn_moment_summary(X_cov,y,learn_type='regression',max_moment=10)
#    embedding_cov.append( lambda x: output[2].predict( np.array([[np.mean(x)]+[stats.moment(x,moment=i) for i in range(2,output[1])]]) )[0] )
#    
#
#df_unit_table = {}
#for i in range(n_paper):
#    paper =  df_paper.loc[i]
#    authors = paper['authors']
#    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
#    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
#    embedded_prestige = np.percentile(prestige_vec,75)
#    embedded_citation = embedding_cov[0](citation_vec)
#    d_paperi = {}
#    d_paperi['quality'] = paper['quality']
#    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
#    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
#    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
#    d_paperi['review'] = paper['review']
#    d_paperi['embedded_prestige'] = embedded_prestige
#    d_paperi['embedded_citation'] = embedded_citation
#    df_unit_table[i] = d_paperi
#
#
#df_unit_table = pd.DataFrame.from_dict(df_unit_table,orient='index')
#
#df_single = df_unit_table.loc[df_unit_table['venue_single-blind'] == 1]
#df_double = df_unit_table.loc[df_unit_table['venue_single-blind'] == 0]
#df_single_treated = df_single.loc[df_single['embedded_prestige']>10]
#df_single_control = df_single.loc[df_single['embedded_prestige']<=10]
#df_double_treated = df_double.loc[df_double['embedded_prestige']>10]
#df_double_control = df_double.loc[df_double['embedded_prestige']<=10]
#
#review_diff = np.mean(df_single['review']) - np.mean(df_double['review'])
#naive_single_ate = np.mean(df_single_treated['review']) - np.mean(df_single_control['review'])
#naive_double_ate = np.mean(df_double_treated['review']) - np.mean(df_double_control['review'])
#
#print('Learned Embedding using Moment Summarization followed by Random Forest',file=fl)
#print(',Mean Difference of Reviews (Single-Double), %f'%(review_diff),file=fl)
#print('Single Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_single_ate),file=fl)
#print('Double Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_double_ate),file=fl)
#
#fig = plt.figure(figsize=(10.5,9.5))
#sns.distplot(df_single_control['review'],hist=False,kde_kws={'shade': True})
#sns.distplot(df_single_treated['review'],hist=False,kde_kws={'shade': True})
#plt.xlim((0,10))
#plt.xlabel('Review Score')
#plt.ylabel('Probability Density Estimate')
#plt.legend(['control','treated'])
#plt.title('Single-Blind')
#fig.savefig('Figures/Learn_MomSum_RF/pdf_single_treated_control_review.png')
#
#fig = plt.figure(figsize=(10.5,9.5))
#sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
#sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
#plt.xlim((0,10))
#plt.xlabel('Review Score')
#plt.ylabel('Probability Density Estimate')
#plt.legend(['control','treated'])
#plt.title('Double-Blind')
#fig.savefig('Figures/Learn_MomSum_RF/pdf_double_treated_control_review.png')
#
#m_s_c = RandomForestRegressor()
#m_s_d = RandomForestRegressor()
#
#m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor','embedded_citation']],df_single_control['review'])
#m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor','embedded_citation']],df_single_treated['review'])
#tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor','embedded_citation']]) - m_s_c.predict(df_double[['quality','venue_impact_factor','embedded_citation']])
#ate_single = np.mean(tau_single)
#mediante_single = np.median(tau_single)
#truth_single = 1.0
#
#m_d_c = RandomForestRegressor()
#m_d_d = RandomForestRegressor()
#
#m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor','embedded_citation']],df_double_control['review'])
#m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor','embedded_citation']],df_double_treated['review'])
#tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor','embedded_citation']]) - m_d_c.predict(df_single[['quality','venue_impact_factor','embedded_citation']])
#ate_double = np.mean(tau_double)
#mediante_double = np.median(tau_double)
#truth_double= 0.0
#
#
#fig = plt.figure(figsize=(10.5,9.5))
#sns.distplot(tau_single,hist=False,kde_kws={'shade': True})
#sns.distplot(tau_double,hist=False,kde_kws={'shade': True})
#plt.axvline(ate_single,color='r',linestyle='--',alpha=0.6)
#plt.axvline(mediante_single,color='g',linestyle='-',alpha=0.6)
#plt.axvline(truth_single, color='b', linestyle='-',alpha=0.6)
#plt.axvline(ate_double,color='y',linestyle='--',alpha=0.6)
#plt.axvline(mediante_double,color='m',linestyle='-',alpha=0.6)
#plt.axvline(truth_double, color='c', linestyle='-',alpha=0.6)
#plt.legend([r'Mean $\tau$ Single-Blind',r'Median $\tau$ Single-Blind',r'True $\tau$ Single-Blind',r'Mean $\tau$ Double-Blind',r'Median $\tau$ Double-Blind',r'True $\tau$ Double-Blind','Single-Blind','Double-Blind'])
#plt.xlabel('Estimated Treatment Effect')
#plt.ylabel('Probability Density Estimate')
#plt.title(r'Single Blind vs Double Blind $\tau$ s')
#fig.savefig('Figures/Learn_MomSum_RF/pdf_single_double_cate.png')
#print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
#print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)
#
#
##---------------------------------------------------------
##Learn Moment Summary Embedding
##---------------------------------------------------------
#
#
#df_unit_table_1 = {}
#n_paper = len(df_paper)
#for i in range(n_paper):
#    paper =  df_paper.loc[i]
#    authors = paper['authors']
#    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
#    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
#    d_paperi = {}
#    d_paperi['quality'] = paper['quality']
#    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
#    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
#    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
#    d_paperi['review'] = paper['review']
#    d_paperi['prestige'] = prestige_vec
#    d_paperi['citation'] = citation_vec
#    df_unit_table_1[i] = d_paperi
#    
#df_unit_table_1 = pd.DataFrame.from_dict(df_unit_table_1,orient='index') 
#embedding_needed_covariates = ['citation']
#embedding_cov = []
#y = np.array(df_unit_table_1['review'])
#for cov in embedding_needed_covariates:
#    X_cov = df_unit_table_1[cov]
#    output = learn_moment_summary(X_cov,y,learn_type='regression',max_moment=10)
#    embedding_cov.append( lambda x: [np.mean(x)]+[stats.moment(x,moment=i) for i in range(2,output[1])] ) 
#    
#
#df_unit_table = {}
#for i in range(n_paper):
#    paper =  df_paper.loc[i]
#    authors = paper['authors']
#    prestige_vec = [ df_inst.loc[df_auth.loc[a]['affiliation']]['prestige'] for a in authors ]
#    citation_vec = [ df_auth.loc[a]['citation'] for a in authors ]
#    embedded_prestige = np.percentile(prestige_vec,75)
#    embedded_citation = embedding_cov[0](citation_vec)
#    d_paperi = {}
#    d_paperi['quality'] = paper['quality']
#    d_paperi['venue_area'] = df_conf.loc[paper['venue']]['area']
#    d_paperi['venue_impact_factor'] = df_conf.loc[paper['venue']]['impact_factor']
#    d_paperi['venue_single-blind'] = df_conf.loc[paper['venue']]['single-blind']
#    d_paperi['review'] = paper['review']
#    d_paperi['embedded_prestige'] = embedded_prestige
#    for k in range(0,len(embedded_citation)):
#        d_paperi['embedded_citation_%d'%(k)] = embedded_citation[k]
#    df_unit_table[i] = d_paperi
#
#
#df_unit_table = pd.DataFrame.from_dict(df_unit_table,orient='index')
#
#df_single = df_unit_table.loc[df_unit_table['venue_single-blind'] == 1]
#df_double = df_unit_table.loc[df_unit_table['venue_single-blind'] == 0]
#df_single_treated = df_single.loc[df_single['embedded_prestige']>10]
#df_single_control = df_single.loc[df_single['embedded_prestige']<=10]
#df_double_treated = df_double.loc[df_double['embedded_prestige']>10]
#df_double_control = df_double.loc[df_double['embedded_prestige']<=10]
#
#review_diff = np.mean(df_single['review']) - np.mean(df_double['review'])
#naive_single_ate = np.mean(df_single_treated['review']) - np.mean(df_single_control['review'])
#naive_double_ate = np.mean(df_double_treated['review']) - np.mean(df_double_control['review'])
#
#print('Learned Embedding using Moment Summarization',file=fl)
#print(',Mean Difference of Reviews (Single-Double), %f'%(review_diff),file=fl)
#print('Single Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_single_ate),file=fl)
#print('Double Blind, Mean Difference of Reviews (Treated-Control), %f'%(naive_double_ate),file=fl)
#
#fig = plt.figure(figsize=(10.5,9.5))
#sns.distplot(df_single_control['review'],hist=False,kde_kws={'shade': True})
#sns.distplot(df_single_treated['review'],hist=False,kde_kws={'shade': True})
#plt.xlim((0,10))
#plt.xlabel('Review Score')
#plt.ylabel('Probability Density Estimate')
#plt.legend(['control','treated'])
#plt.title('Single-Blind')
#fig.savefig('Figures/Learn_MomSum/pdf_single_treated_control_review.png')
#
#fig = plt.figure(figsize=(10.5,9.5))
#sns.distplot(df_double_control['review'],hist=False,kde_kws={'shade': True})
#sns.distplot(df_double_treated['review'],hist=False,kde_kws={'shade': True})
#plt.xlim((0,10))
#plt.xlabel('Review Score')
#plt.ylabel('Probability Density Estimate')
#plt.legend(['control','treated'])
#plt.title('Double-Blind')
#fig.savefig('Figures/Learn_MomSum/pdf_double_treated_control_review.png')
#
#m_s_c = RandomForestRegressor()
#m_s_d = RandomForestRegressor()
#
#m_s_c = m_s_c.fit(df_single_control[['quality','venue_impact_factor']+['embedded_citation_'+str(k) for k in range(0,len(embedded_citation))]],df_single_control['review'])
#m_s_d = m_s_d.fit(df_single_treated[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]],df_single_treated['review'])
#tau_single = m_s_d.predict(df_double[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]]) - m_s_c.predict(df_double[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]])
#ate_single = np.mean(tau_single)
#mediante_single = np.median(tau_single)
#truth_single = 1.0
#
#m_d_c = RandomForestRegressor()
#m_d_d = RandomForestRegressor()
#
#m_d_c = m_d_c.fit(df_double_control[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))] ],df_double_control['review'])
#m_d_d = m_d_d.fit(df_double_treated[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))] ],df_double_treated['review'])
#tau_double = m_d_d.predict(df_single[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))] ]) - m_d_c.predict(df_single[['quality','venue_impact_factor']+['embedded_citation_%d'%(k) for k in range(0,len(embedded_citation))]])
#ate_double = np.mean(tau_double)
#mediante_double = np.median(tau_double)
#truth_double= 0.0
#
#
#fig = plt.figure(figsize=(10.5,9.5))
#sns.distplot(tau_single,hist=False,kde_kws={'shade': True})
#sns.distplot(tau_double,hist=False,kde_kws={'shade': True})
#plt.axvline(ate_single,color='r',linestyle='--',alpha=0.6)
#plt.axvline(mediante_single,color='g',linestyle='-',alpha=0.6)
#plt.axvline(truth_single, color='b', linestyle='-',alpha=0.6)
#plt.axvline(ate_double,color='y',linestyle='--',alpha=0.6)
#plt.axvline(mediante_double,color='m',linestyle='-',alpha=0.6)
#plt.axvline(truth_double, color='c', linestyle='-',alpha=0.6)
#plt.legend([r'Mean $\tau$ Single-Blind',r'Median $\tau$ Single-Blind',r'True $\tau$ Single-Blind',r'Mean $\tau$ Double-Blind',r'Median $\tau$ Double-Blind',r'True $\tau$ Double-Blind','Single-Blind','Double-Blind'])
#plt.xlabel('Estimated Treatment Effect')
#plt.ylabel('Probability Density Estimate')
#plt.title(r'Single Blind vs Double Blind $\tau$ s')
#fig.savefig('Figures/Learn_MomSum/pdf_single_double_cate.png')
#print('Single-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_single,mediante_single,truth_single),file=fl)
#print('Double-Blind \nATE, %f \nMedian, %f \nTrue TE, %f'%(ate_double,mediante_double,truth_double),file=fl)
#fl.close()
