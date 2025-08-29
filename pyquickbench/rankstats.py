import itertools
import math
import numpy as np

from .cython.rankstats import (
    left_lehmer                     ,
    from_left_lehmer                ,
    score_to_perm_count_inner_loop  ,
)
    
def score_to_perm_count(l):
    
    nvec = len(l)

    shapes = [l[i].shape[0] for i in range(nvec)]
    
    prod = 1
    for i in range(nvec):
        prod *= shapes[i]

    if np.iinfo(np.intp).max < prod:
        raise ValueError("Too many observations in vectors")

    nelem_tot = sum(shapes)
    
    cum_shapes = np.zeros(nvec, dtype=np.intp)
    cum_shapes[0] = shapes[0]
    for i in range(nvec-1):
        cum_shapes[i+1] = cum_shapes[i] + shapes[i+1]
    
    u = np.concatenate(l)
    idx_sort = np.argsort(u)
    idx_sorted_to_ivec = np.searchsorted(cum_shapes, idx_sort, side='right')

    idx_sorted_to_ivec_compressed = []
    idx_sorted_to_ivec_len = []
    
    start = 0
    end = 0
    while end < nelem_tot:

        for end in range(start+1, nelem_tot):
            if idx_sorted_to_ivec[start] != idx_sorted_to_ivec[end]:
                break
        else:
            end = nelem_tot
            
        idx_sorted_to_ivec_compressed.append(idx_sorted_to_ivec[start])
        idx_sorted_to_ivec_len.append(end-start)
        
        start = end

    idx_sorted_to_ivec_compressed = np.array(idx_sorted_to_ivec_compressed, dtype=np.intp)
    idx_sorted_to_ivec_len_arr = np.array(idx_sorted_to_ivec_len, dtype=np.intp)
    nelem_reduced = idx_sorted_to_ivec_len_arr.shape[0]

    ivec_to_idx_sorted_compressed = [[] for i in range(nvec)]
    
    for i in range(nelem_reduced):
        ivec_to_idx_sorted_compressed[idx_sorted_to_ivec_compressed[i]].append(i)

    for i in range(nvec):
        ivec_to_idx_sorted_compressed[i] = np.array(ivec_to_idx_sorted_compressed[i])

    return score_to_perm_count_inner_loop(ivec_to_idx_sorted_compressed, idx_sorted_to_ivec_len_arr)

def score_to_perm_count_brute_force(l):
    
    nvec = len(l)
    fac = math.factorial(nvec)

    prod = 1
    for i in range(nvec):
        prod *= l[i].shape[0]

    if np.iinfo(np.intp).max < prod:
        raise ValueError("Too many observations in vectors")
    
    res = np.zeros(fac, dtype=np.intp)   
    vals = np.empty(nvec, dtype=np.float64)
    
    ranges = [range(l[i].shape[0]) for i in range(nvec)]
    
    for I in itertools.product(*ranges):
        
        for i in range(nvec):
            vals[i] = l[i][I[i]]
        
        perm = np.argsort(vals)
        i = left_lehmer(perm)
        res[i] += 1
        
    return res

def score_to_partial_order_count(k, l, opt = "opt"):
    
    nvec = len(l)
    nfac = math.factorial(k)
    ncomb = math.comb(nvec, k)

    if np.iinfo(np.intp).max < nfac:
        raise ValueError("Too many vectors")

    res = np.zeros((ncomb, nfac), dtype=np.intp)    
    
    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):
        
        ll = [l[c] for c in comb]
        
        if opt == "opt":
            res[icomb,:] = score_to_perm_count(ll)
        elif opt == "brute_force":
            res[icomb,:] = score_to_perm_count_brute_force(ll)
        else:
            raise ValueError('Unknown backend')
        
    return res

def find_nvec_k_from_order_count_shape(order_count, kmax=100, nvec_max = 20):
    
    # Find nvec, k such that factorial(k) == order_count.shape[1] and comb(k,nvec) == order_count.shape[0]
    
    nsets = order_count.shape[0]
    nopt_per_set = order_count.shape[1]
    
    kfac = 1
    for k in range(1,kmax):
        kfac *= k
        if kfac == nopt_per_set:
            break
    else:
        raise ValueError("Could not determine k")
    
    c_arr = np.ones(nvec_max, dtype=np.intp)
    for nvec in range(1,k):
        for kk in range(nvec-1,0,-1):
            c_arr[kk] += c_arr[kk-1] 

    for nvec in range(k,nvec_max):
        for kk in range(nvec-1,0,-1):
            c_arr[kk] += c_arr[kk-1] 

        if c_arr[k] == nsets:
            break
    else:
        raise ValueError("Could not determine nvec")
    
    return nvec, k

def condorcet_top_order(order_count, minimize=False):
    
    nvec, k = find_nvec_k_from_order_count_shape(order_count)
    
    if (k != 2):
        raise ValueError(f'Expected pairwise comparison data. Received {k}-wise comparison data')
    
    if minimize:
        i_opt = 0
    else:
        i_opt = 1
        
    available_options = list(range(nvec))
    
    res = np.full(nvec, -1, dtype=np.intp)
    
    for i_round in range(nvec):
        
        nrem_vec = nvec-i_round
            
        n_wins = np.zeros(nrem_vec, dtype=np.intp)
        
        for iset, comb in enumerate(itertools.combinations(range(nrem_vec), 2)):

            if order_count[iset, i_opt] > order_count[iset, 1-i_opt]:
                n_wins[available_options[comb[0]]] += 1
            else:
                n_wins[available_options[comb[1]]] += 1
                
        most_wins = np.argmax(n_wins)
        
        if n_wins[most_wins] == (nrem_vec-1):
            res[nrem_vec-1] = available_options[most_wins]
            available_options.pop(most_wins)
        else:
            break
        
    return res


def build_sinkhorn_problem(order_count, minimize=False):
    
    nsets = order_count.shape[0]
    nopt_per_set = order_count.shape[1]

    nvec, k = find_nvec_k_from_order_count_shape(order_count)
    
    nopts = nvec
    
    if minimize:
        i_opt = 0
    else:
        i_opt = k-1

    p_int = np.zeros(nsets, dtype=order_count.dtype)    
    q_int = np.zeros(nopts, dtype=order_count.dtype)
    
    total_sum = 0
    
    A = np.full((nsets, nopts), np.inf, dtype=np.float64)
    
    for iset, comb in enumerate(itertools.combinations(range(nvec), k)):

        for jperm in range(nopt_per_set):
            
            perm = from_left_lehmer(jperm, k)
            
            val = order_count[iset, jperm]
            
            total_sum += val
            p_int[iset] += val
            q_int[comb[perm[i_opt]]] += val
            
            for j in comb:
                A[iset, j] = 0.
        
    p = np.empty(nsets, dtype=np.float64)    
    q = np.empty(nopts, dtype=np.float64)
        
    for i in range(nsets):
        p[i] = p_int[i] / total_sum        
    for i in range(nopts):
        q[i] = q_int[i] / total_sum
                
    return A, p, q
            
            
        
    
    