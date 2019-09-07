#!/usr/bin/env python3

import json
from collections import defaultdict

def collaboration(paper):
    domains = set()

    for a in paper['authorids']:
        d = a.split("@")[-1]
        if d == "gmail.com" or d == "yahoo.com":
            continue
        domains.add(d)

    return len(domains) > 1

with open("../openreview-dataset/results/papers.json", "r") as f:
    papers = json.load(f)

with open("../openreview-dataset/results/reviews.json", "r") as f:
    reviews = json.load(f)

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
