import random

def generate(entities, parameters):
    """
        Generate the relationships for the provided entities.
    """
    R = {}
    R["s_s"] = generate_friends(entities["s"], parameters.P_f)
    R["s_c"] = generate_registration(entities["s"], entities["c"], parameters.P_r)
    return R

    
def generate_friends(students, p_f):
    """
        Generate friendship relationship, an Erdős–Rényi model.
        Any two students are friends with probability p,
        No student can be a friend of themselves. All friendship
        is bidirectional.
    """
    friendships = dict()
    for s1 in students:
        friendships.setdefault(s1,[])
        for s2 in students:
            if s1 != s2 and random.random() < p_f:
                friendships[s1].append(s2)
                friendships.setdefault(s2,[]).append(s1)
    return friendships

    
def generate_registration(students, courses, p_r):
    """
        Course registration is a random bipartite
        graph between students and courses. Any
        edge exists with probability p_r.
    """
    registration = dict()
    for s in students:
        registration[s] = []
        for c in courses:
            if random.random() < p_r:
                registration[s].append(c)
    return registration
