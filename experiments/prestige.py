#!/usr/bin/env python
from common import load_dataset, prestigious, h_index

import numpy as np

all_authors, papers, reviews, confs = load_dataset()

flat = []
for p in papers:
    conf_blind = confs[p["conf"]]["blind"]
    authors = [all_authors[i] for i in filter(None, p["author_keys"])]
    for a in authors:
        # experience is range of publication years
        try:
            pub_info = a["scopus"]["_json"]["author-profile"]["publication-range"]
            experience = int(pub_info["@end"]) - int(pub_info["@start"])
        except (KeyError, TypeError):
            experience = 0

        prestige = prestigious(a["world_rank"])
        impact = h_index(a)
        decision = "reject" not in p["decision"].lower()
        row = [impact, prestige, experience, conf_blind, decision]
        flat.append(row)

flat = np.array(flat)
print(np.mean(flat, axis=0))
