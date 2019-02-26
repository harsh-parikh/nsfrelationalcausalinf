#!/usr/bin/env python3

import queue


"""
Takes a rooted, directed graph as an adjacency list and returns another
adjacency list corresponding to a breadth-first search tree.
"""
def bfs_search_tree(subject, adjacency):
    search_tree = dict()
    known = {subject}
    dests = queue.Queue()
    dests.put(subject)
    
    while not dests.empty():
        upcoming = dests.get()
        children = {c for c in adjacency.get(upcoming, {}) if c not in known}
        for child in children:
            known.add(child)
            dests.put(child)
            if upcoming in search_tree:
                search_tree[upcoming].add(child)
            else:
                search_tree[upcoming] = {child}               
 
    # include an empty entry for vertices with no out-edges, just in case
    for i in known - set(search_tree):
        search_tree[i] = {}

    return search_tree


def simple_test():
    subject = "A"
    connections = {
        "A": {"B", "C"},
        "B": {"C"},
        "C": {"C", "D"}
    }

    print(bfs_search_tree(subject, connections))
