import random

from sklearn.preprocessing import Normalizer
from scipy.special import expit
from numpy.random import normal
import progressbar
import numpy as np

from synthetic import summarize

def generate(entities, relations, param):
    A = {}
    A["d"] = generate_difficulty(entities, param)
    A["att"] = generate_attendence(entities, relations, A, param)

    
def generate_difficulty(entities, param):
    return [random.random() < param.P_d for c in entities["c"]]

    
def generate_attendence(entities, relations, attributes, param):
    # initialize lecture attendence completely randomly
    attend = dict()
    for s in relations["s_c"]:
        for c in relations["s_c"][s]:
            attend[(s, c)] = random.choice([0, 1])
    
    # Gibbs sampling
    for t in progressbar.progressbar(range(param.time)):
        print(f"\n {np.mean(np.fromiter(attend.values(), dtype=float)) * 100}")
        
        for s in relations["s_c"]:
            for c in relations["s_c"][s]:            
                prev_attend = 0.3 * attend[(s, c)]
                phi_skill = 0
                friends_attend = summarize.beta_9 * friends_attendence(s, c, relations, attend)
                diff = attributes["d"][c]
                noise = normal()
                attend[(s, c)] = prev_attend + phi_skill + friends_attend + diff + noise
    
        all_values = np.fromiter(attend.values(), dtype=float).reshape(-1, 1)
        normalizer = Normalizer()
        normalizer.fit(all_values)
        for k in attend:
            v = np.array(attend[k]).reshape(-1, 1)
            attend[k] = random.random()<expit(normalizer.transform(v))
        
  
def friends_attendence(ego, course, relations, attendence):
    """
        Return a summarization of the attendence of the friends
        of the ego.
    """
    f = relations["s_s"][ego]
    f = [x for x in filter(lambda s : course in relations["s_c"][s], f)]
    f_att = [attendence[(f, course)] for f in f]
    return summarize.phi_FLA(f_att)