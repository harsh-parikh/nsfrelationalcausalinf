#!/usr/bin/env python3

import json
from collections import defaultdict

import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression
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

with open("../openreview-dataset/results/authors.json", "r") as f:
    authors = json.load(f)

with open("../openreview-dataset/results/papers.json", "r") as f:
    papers = json.load(f)

with open("../openreview-dataset/results/reviews.json", "r") as f:
    reviews = json.load(f)

with open("../openreview-dataset/results/confs.json", "r") as f:
    confs = json.load(f)

hash_to_review = defaultdict(list)
for r in reviews:
    hash_to_review[r['paperhash']].append(r)

data = defaultdict(dict)
target = {}

for p in papers:
    id = p['paperhash']
    target[id] = 'reject' in p['decision'].lower()
    data[id]['collab'] = collaboration(p)
    if not hash_to_review[id]:
        data[id]['rating_avg'] = 0.5
        data[id]['conf_avg'] = 0
    else:
        data[id]['rating_avg'] = np.mean([r['norm_rating'] for r in hash_to_review[id]])
        data[id]['conf_avg'] = np.mean([r['norm_conf'] for r in hash_to_review[id]])
    data[id]['conf_rigor'] = confs[p['conf']]['rigor']

data = np.array([process(data[k]) for k in sorted(data)])
target = np.array(process(target))

lr = LogisticRegression(solver='liblinear')
scores = cross_val_score(lr, data, target, cv=5)
print("Accuracy: %0.2f (± %0.2f)" % (scores.mean(), scores.std() * 2))
