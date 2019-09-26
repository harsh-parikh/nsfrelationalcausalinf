#!/usr/bin/env python
from common import load_dataset, publishing_years, prestigious, h_index

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
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

        rs = reviews[p["paper_id"]]
        score = np.mean([r["norm_rating"] for r in rs]) if rs else 0

        # target
        decision = "reject" not in p["decision"].lower()

        # insert the row
        row = [
            # impact,
            # prestige,
            # experience,
            co_prestige > 0, 
            co_impact,
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
flat = np.vstack((flat_single, flat_double))

binner = KBinsDiscretizer(n_bins=[3, 3], encode="ordinal", strategy="kmeans")
binner.fit(flat[:, 1:-1])

flat_single[:, 1:-1] = binner.transform(flat_single[:, 1:-1])
flat_double[:, 1:-1] = binner.transform(flat_double[:, 1:-1])

count_p = len([r for r in flat if r[0]])
count_np = len(flat) - count_p

print("==== Matching ====")
result_sb = 0
for target_i in range(3):
    for target_e in range(3):
        p_outcomes = []
        np_outcomes = []

        for r in flat_single:
            if r[0] == 0 and r[1] == target_i and r[2] == target_e:
                np_outcomes.append(r[3])
            if r[0] == 1 and r[1] == target_i and r[2] == target_e:
                p_outcomes.append(r[3])

        delta = np.mean(p_outcomes) - np.mean(np_outcomes)
        if not np.isnan(delta):
            count = len(p_outcomes) + len(np_outcomes)
            adjustment = count / len(flat_single)
            result_sb += delta * adjustment
        else:
            pass
print("single blind: ", result_sb)

result_db = 0
for target_i in range(3):
    for target_e in range(3):
        p_outcomes = []
        np_outcomes = []

        for r in flat_double:
            if r[0] == 0 and r[1] == target_i and r[2] == target_e:
                np_outcomes.append(r[3])
            if r[0] == 1 and r[1] == target_i and r[2] == target_e:
                p_outcomes.append(r[3])

        delta = np.mean(p_outcomes) - np.mean(np_outcomes)
        if not np.isnan(delta):
            count = len(p_outcomes) + len(np_outcomes)
            adjustment = count / len(flat_double)
            result_db += delta * adjustment
        else:
            pass
print("double blind: ", result_db)

print("difference:", result_db - result_sb)
# plt.hist(X_single[:, 1], alpha=0.2)
# plt.hist(X_double[:, 1], alpha=0.2)
# plt.show()

print("===== Naive Averaging ===")
print("single blind, prestigious: ", np.mean([r[3] for r in flat_single if r[0]]))
print("single blind, not prestigious: ", np.mean([r[3] for r in flat_single if not r[0]]))
print("single blind, p - np: ", np.mean([r[3] for r in flat_single if r[0]]) - np.mean([r[3] for r in flat_single if not r[0]]))
print("double blind, prestigious: ", np.mean([r[3] for r in flat_double if r[0]]))
print("double blind, not prestigious: ", np.mean([r[3] for r in flat_double if not r[0]]))
print("double blind, p - np: ", np.mean([r[3] for r in flat_double if r[0]]) - np.mean([r[3] for r in flat_double if not r[0]]))
