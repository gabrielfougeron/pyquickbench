import itertools
import math
import numpy as np

from .cython.rankstats import *

def score_to_partial_order_count(k, l):
    
    nvec = len(l)
    nfac = math.factorial(k)
    ncomb = math.comb(nvec, k)

    res = np.zeros((ncomb, nfac), dtype=np.intp)    
    
    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):
        
        ll = [l[c] for c in comb]
        res[icomb,:] = score_to_perm_count(ll)
        
    return res

def score_to_partial_order_count_brute_force(k, l):
    
    nvec = len(l)
    nfac = math.factorial(k)
    ncomb = math.comb(nvec, k)
    
    res = np.zeros((ncomb, nfac), dtype=np.intp)    
    vals = np.empty(k, dtype=np.float64)
    
    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):
    
        ranges = [range(l[i].shape[0]) for i in comb]
        
        for I in itertools.product(*ranges):
            
            for i in range(k):
                vals[i] = l[comb[i]][I[i]]
            
            perm = np.argsort(vals)
            i = left_lehmer(perm)
            res[icomb, i] += 1
            
    return res

def build_sinkhorn_problem(k, l):
    
    nvec = len(l)
    order_count = score_to_partial_order_count(k, l)
    
    ncomb = order_count.shape[0]
    nfac = order_count.shape[1]
    