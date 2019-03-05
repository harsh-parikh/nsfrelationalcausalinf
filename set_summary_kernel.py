#input subject-id-column (sid) , summarize-columns
import numpy as np
import scipy
from sklearn.neighbors.kde import KernelDensity

def subset(D,sid,i):
    Di = 0
    return Di

def summarize_set(D,sid,summary_col_array):
    #for each element of sid, form a set and summarize the set using KDE.
    sidarray = unique(D[:,sid])
    for i in sidarray:
        Di = subset(D,sid,i)
        Xi = subcol(Di,summary_col_array)