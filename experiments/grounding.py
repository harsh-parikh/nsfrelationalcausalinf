#!/usr/bin/env python3

import json
from collections import defaultdict


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


with open("../openreview-dataset/results/papers.json", "r") as f:
    papers = json.load(f)

with open("../openreview-dataset/results/reviews.json", "r") as f:
    reviews = json.load(f)

with open("../openreview-dataset/results/confs.json", "r") as f:
    confs = json.load(f)

# maps unit (symbol, value) to set of incoming edges
grounding = defaultdict(set)

for r in reviews:
    # Accept[P] ⃪ Score[P, R] where Reviewed(R, P).
    grounding[("accept", r["paperhash"])].add(("score", r["norm_rating"]))

    # Accept[P] ⃪ Confidence[P, R] where Reviewed(R, P).
    grounding[("accept", r["paperhash"])].add(("confidence", r["norm_conf"]))

for p in papers:
    # Accept[P] ⃪ Is_Collab[P].
    grounding[("accept", p["paperhash"])].add(("is_collab", collaboration(p)))

    # Accept[P] ⃪ Rigor[C] where Submitted(P, C).
    grounding[("accept", p["paperhash"])].add(("rigor", confs[p["conf"]]["rigor"]))
