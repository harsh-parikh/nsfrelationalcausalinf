import random
import json

'''
    Helper class for argparse
'''
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
    def __str__(self):
        return f"{self.start} - {self.end}"

'''
    Returns a list of edges corresponding to the bipartite graph
    connecting the two sets of vertices `A` and `B`. For each a ∈ A,
    b ∈ B, the edge (a, b) exists with probablity `p`.
    If an edge c ∈ A, B then the edge (c, c) is considered
    only if self_loops is true.
'''
def bipartite_graph(A, B, p, self_loops=True):
    result = set()
    for a in A:
        for b in B:
            if (self_loops or a != b) and random.random() < p:
                result.add((a, b))
    return result

'''
    Helper to encode Python sets as JSON lists
'''
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
       if isinstance(obj, set):
          return list(obj)
          return json.JSONEncoder.default(self, obj)
