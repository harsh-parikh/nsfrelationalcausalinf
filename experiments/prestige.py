#!/usr/bin/env python
from common import load_dataset, publishing_years, prestigious, h_index

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

all_authors, papers, reviews, confs = load_dataset()

flat_single = []
flat_double = []
for p in papers:
    conf_blind = confs[p["conf"]]["blind"]
    authors = [all_authors[i] for i in filter(None, p["author_keys"])]
    for a in authors:
        # direct effects
        experience = publishing_years(a)
        prestige = prestigious(a)
        impact = h_index(a)

        # relational effects
        coauthors = [c for c in authors]
        if coauthors:
            co_impact = np.mean([h_index(c) for c in coauthors])
            co_prestige = np.mean([prestigious(c) for c in coauthors])
            co_experience = np.mean([publishing_years(c) for c in coauthors])
        else:
            co_impact = co_prestige = co_experience = 0

        rs = reviews[p['paper_id']]
        score = np.mean([r['norm_rating'] for r in rs]) if rs else 0

        # target
        decision = "reject" not in p["decision"].lower()

        # insert the row
        row = [
            #impact,
            #prestige,
            #experience,
            co_impact,
            co_prestige,
            co_experience,
            # score,
            decision,
        ]
        if conf_blind:
            flat_double.append(row)
        else:
            flat_single.append(row)
        break

flat_single = np.array(flat_single)
flat_double = np.array(flat_double)

X_single = flat_single[:, :-1]
y_single = np.ravel(flat_single[:, -1:])

X_double = flat_double[:, :-1]
y_double = np.ravel(flat_double[:, -1:])

X_single_train, X_single_test, y_single_train, y_single_test = train_test_split(X_single, y_single, test_size=0.25)
X_double_train, X_double_test, y_double_train, y_double_test = train_test_split(X_double, y_double, test_size=0.25)

single_model = LogisticRegression()
single_model.fit(X_single_train, y_single_train)

double_model = LogisticRegression()
double_model.fit(X_double_train, y_double_train)

# estimation
print(double_model.coef_)
print(single_model.coef_)
print("====")
print(double_model.coef_ - single_model.coef_)

plt.hist(X_single[:, 1], alpha=0.2)
plt.hist(X_double[:, 1], alpha=0.2)
plt.show()
