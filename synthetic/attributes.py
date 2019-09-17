import random

from sklearn.preprocessing import Normalizer
from scipy.special import expit
from numpy.random import normal
import progressbar
import numpy as np

from synthetic import summarize

def generate(entities, relations, param):
    s_count = len(entities["s"])
    p_count = len(entities["p"])
    c_count = len(entities["c"])
    
    A = {}
    
    # student attributes
    A["s_age"] = np.random.randint(low=17, high=23, size=S_count)
    A["s_gender"] = random.random() < param.P_female
    ## TODO find truncnorm distribution once online again
    A["s_gpa"] = np.random.normal(3, 1, S_count)
    A["s_major"] = np.random.multinomial(1, [1/3, 1/3, 1/6, 1/6], s_count)
    
    # professor attributes
    A["p_age"] = 0  ## TODO truncnorm
    A["p_rating"] = 0
    A["p_exp"] = 0 
    A["p_dept"] = np.random.multinomial(1, [1/2, 1/3, 1/12, 1/12], p_count)
    
    # course attributes
    A["c_level"] = generate_difficulty(entities, param)
    A["c_dept"] = np.random.multinomial(1, [1/2, 1/3, 1/12, 1/12], c_count)
    
    # composite attributes
    A["att"] = generate_attendence(entities, relations, A, param)

    
def generate_attendence(entities, relations, attributes, param):
    # initialize lecture attendence completely randomly
    attend = dict()
    for s in relations["s_c"]:
        for c in relations["s_c"][s]:
            attend[(s, c)] = random.choice([0, 1])
    
    # Gibbs sampling
    for t in range(param.time):
        print(f"{np.mean(np.stack(attend.values())) * 100}")
        for s in relations["s_c"]:
            for c in relations["s_c"][s]:            
                prev_attend = attend[(s, c)]
                phi_skill = -attributes["a"][s]
                friends_attend = friends_attendence(s, c, relations, attend)
                
                diff = int(attributes["d"][c])
                noise = normal()
                v = sum([prev_attend, phi_skill, friends_attend,
                              diff, noise])
                attend[(s, c)] = random.random() < expit(v)
  

def friends_attendence(ego, course, relations, attendence):
    """
        Return a summarization of the attendence of the friends
        of the ego.
    """
    f = relations["s_s"][ego]
    f = [x for x in filter(lambda s : course in relations["s_c"][s], f)]
    f_att = [attendence[(f, course)] for f in f]
    return summarize.phi_FLA(f_att)
