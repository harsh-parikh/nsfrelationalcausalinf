#!/usr/bin/env python
from common import load_dataset, publishing_years, prestigious, h_index

import numpy as np
from sklearn.model_selection import cross_val_score

all_authors, papers, reviews, confs = load_dataset()

flat = []
for p in papers:
    conf_blind = confs[p["conf"]]["blind"]
    authors = [all_authors[i] for i in filter(None, p["author_keys"])]
    for a in authors:
        # direct effects
        experience = publishing_years(a)
        prestige = prestigious(a)
        impact = h_index(a)

        # relational effects
        coauthors = [c for c in authors if c != a]
        if coauthors:
            co_impact = np.median([h_index(c) for c in coauthors])
            co_prestige = np.max([prestigious(c) for c in coauthors])
            co_experience = np.max([publishing_years(c) for c in coauthors])
        else:
            co_impact = co_prestige = co_experience = 0

        # target
        decision = "reject" not in p["decision"].lower()

        # insert the row
        row = [
            impact,
            prestige,
            experience,
            co_impact,
            co_prestige,
            co_experience,
            conf_blind,
            decision,
        ]
        flat.append(row)

flat = np.array(flat)
print(np.mean(flat, axis=0))

from bartpy.sklearnmodel import SklearnModel
model = SklearnModel() # Use default parameters
X = flat[:, :-1]
y = np.ravel(flat[:, -1:])

scores = cross_val_score(model, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
