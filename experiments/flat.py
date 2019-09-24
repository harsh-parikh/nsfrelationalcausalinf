#!/usr/bin/env python3

import json
from collections import defaultdict

import numpy as np
import sklearn

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score

def collaboration(paper):
    '''Decide if paper is a collaboration between multiple institutions.

    Do this by looking at email domains, ignoring commercial providers.
    '''
    ignore = {"gmail", "yahoo", "hotmail"}
    domains = set()

    for a in paper['authorids']:
        d = a.split("@")[-1]
        if any(x in d for x in ignore):
            continue
        domains.add(d)

    return int(len(domains) > 1)


def process(d):
    return [d[k] for k in sorted(d)]

def citations(a_id):
    try:
        return int(authors[a_id]['scopus']['_json']['coredata']['citation-count'])
    except (KeyError, TypeError):
        return 0

with open("../openreview-dataset/results/authors.json", "r") as f:
    authors = json.load(f)

with open("../openreview-dataset/results/papers.json", "r") as f:
    papers = json.load(f)

with open("../openreview-dataset/results/reviews.json", "r") as f:
    reviews = json.load(f)

with open("../openreview-dataset/results/confs.json", "r") as f:
    confs = json.load(f)

data = defaultdict(dict)
target = {}

conf_hash = {pair[1]: pair[0] for pair in enumerate(confs.keys())}

for p in papers:
    id = p['paper_id']
    if id in reviews:
        target[id] = np.mean([r['norm_rating'] for r in reviews[id]])
    
        # target[id] = 'reject' in p['decision'].lower()
    
        # data[id]['collab'] = collaboration(p)
    
        """
        if id not in reviews:
            data[id]['rating_avg'] = 0.5
            data[id]['conf_avg'] = 0
        else:
            data[id]['rating_avg'] = np.mean([r['norm_rating'] for r in reviews[id]])
            data[id]['conf_avg'] = np.mean([r['norm_conf'] for r in reviews[id]])
        """
            
        # data[id]['conf_rigor'] = confs[p['conf']]['rigor']
        data[id]['popularity_avg'] = np.mean([citations(a_id) for a_id in p['author_keys']])
        data[id]['blind'] = confs[p['conf']]['blind']
        # data[id]['workshop'] = confs[p['conf']]['workshop']
        data[id]['conf_id'] = conf_hash[p['conf']]

data = np.array([process(data[k]) for k in sorted(data)])
target = np.array(process(target))

lr = LinearRegression()
scores = cross_val_score(lr, data, target, cv=5)
print("Accuracy: %0.2f (± %0.2f)" % (scores.mean(), scores.std() * 2))

keys = list(range(0,len(confs.keys())))
conf_keys = list(confs.keys())

acc = np.zeros((len(confs.keys()),))
blind = np.zeros((len(confs.keys()),))
for i in range(0,len(confs.keys())):
    d_temp = []
    for j in range(0,len(data)):
        if data[j,1] == i:
            d_temp.append(list(data[j,:])+[target[j]])
    if len(d_temp)>0:
        d_temp = np.array(d_temp)
        lr = RFR()
        lr = lr.fit(np.array(d_temp)[:,2].reshape(-1,1), np.array(d_temp)[:,3])
        scores = lr.score( np.array(d_temp)[:,2].reshape(-1,1), np.array(d_temp)[:,3])#, cv=5)            
        acc[i] = 1-scores
        blind[i] = np.mean(d_temp[:,0])
        fig = plt.figure(figsize=(8.75,7))
        plt.scatter(np.array(d_temp)[:,2], np.array(d_temp)[:,3])
        plt.title('Conference %s'%(conf_keys[i]))
        fig.savefig('output/status_review_conference_%d.png'%(keys[i]))
    
#plt.scatter(range(0,len(acc)),acc,c=blind)
collector_1 = []
label_0 = []
collector_0 = []
label_1 = []
keys = list(range(0,len(confs.keys())))
for i in range(0,len(acc)):
    if blind[i] == 1.0:
        collector_1.append(acc[i])
        label_1.append(keys[i])
    else:
        collector_0.append(acc[i])
        label_0.append(keys[i])
        
stupid_ATE = (np.mean(collector_1) - np.mean(collector_0))
fig = plt.figure(figsize=(8.75,7))
position_0 = np.random.normal(1,0.05,size=len(collector_0))
position_1 = np.random.normal(2,0.05,size=len(collector_1))
plt.scatter(position_0,collector_0,alpha=0.4,s=150)
plt.scatter(position_1,collector_1,alpha=0.4,s=150)
plt.violinplot(collector_0,positions=[1])
plt.violinplot(collector_1,positions=[2])
for i in range(len(label_0)):
    plt.annotate(label_0[i],(position_0[i],collector_0[i]))
for i in range(len(label_1)):
    plt.annotate(label_1[i],(position_1[i],collector_1[i]))
plt.title('Fairness vs Review')
fig.savefig('output/Fairness_Review.png')

print(stupid_ATE)
