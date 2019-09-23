#!/usr/bin/env python
import json

import numpy as np

# universities ranked higher than this are prestigious
PRESTIGE_CUTOFF = 40

with open("../openreview-dataset/results/authors.json", "r") as f:
    all_authors = json.load(f)

with open("../openreview-dataset/results/papers.json", "r") as f:
    papers = json.load(f)

with open("../openreview-dataset/results/reviews.json", "r") as f:
    reviews = json.load(f)

with open("../openreview-dataset/results/confs.json", "r") as f:
    confs = json.load(f)


def prestigious(ranking):
    # "corp" means one of few prestigious corporations (e.g. MSR)
    if ranking == "corp":
        return True

    # either low-ranking corporation or university that is not in top 2000
    if ranking == "":
        return False

    # rankings past 100 are in the form "150-200".
    if "-" in ranking:
        return False

    return int(ranking) < 40


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
