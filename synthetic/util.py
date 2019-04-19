import random
import json


class Range(object):
    """
        Helper class for argparse
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __str__(self):
        return f"{self.start} - {self.end}"


def bipartite_graph(A: set, B: set, p: float, self_loops=True):
    """
        Returns a list of edges corresponding to the bipartite graph
        connecting the two sets of vertices `A` and `B`. For each a ∈ A,
        b ∈ B, the edge (a, b) exists with probablity `p`.
        If an edge c ∈ A, B then the edge (c, c) is considered
        only if self_loops is true.
    """
    result = set()
    for a in A:
        for b in B:
            if (self_loops or a != b) and random.random() < p:
                result.add((a, b))
    return result


class SetEncoder(json.JSONEncoder):
    """
        Helper to encode Python sets as JSON lists
    """
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


