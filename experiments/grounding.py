#!/usr/bin/env python3

import json
from collections import defaultdict

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
