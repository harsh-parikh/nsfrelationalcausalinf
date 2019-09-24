#!/usr/bin/env python
from common import prestigious, load_dataset

import numpy as np

# universities ranked higher than this are prestigious
PRESTIGE_CUTOFF = 40

all_authors, papers, reviews, confs = load_dataset()

flat = []
for p in papers:
    authors = [all_authors[i] for i in filter(None, p["author_keys"])]
    for a in authors:
        # experience is range of publication years
        try:
            pub_info = a["scopus"]["_json"]["author-profile"]["publication-range"]
            experience = int(pub_info["@end"]) - int(pub_info["@start"])
        except (KeyError, TypeError):
            experience = 0

        prestige = prestigious(a["world_rank"])
        decision = "reject" not in p["decision"].lower()
        row = [prestige, experience, decision]
        flat.append(row)

flat = np.array(flat)
print(flat)
