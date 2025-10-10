import collections
import itertools
import math
import numpy as np
import scipy

from .cython.rankstats import (
    left_lehmer                                 ,
    from_left_lehmer                            ,
    rank_combination                            ,
    unrank_combination                          ,
    exhaustive_score_to_best_count_inner_loop   ,
    exhaustive_score_to_perm_count_inner_loop   ,
    montecarlo_score_to_best_count              ,
    montecarlo_score_to_perm_count              ,
    KendallTauDistance                          ,
    KendallTauRankCorrelation                   ,
    find_nvec_k_from_best_count_shape           ,
    find_nvec_k_from_order_count_shape          ,
    order_count_to_best_count                   ,
    build_sinkhorn_rhs                          ,
    build_sinkhorn_rhs_new                      ,
)

from .cython.sinkhorn import (
    sinkhorn_knopp                              ,
    uv_to_loguv                                 ,
)

def find_nvec_k(nsets, nopt_per_set, kmax = 100, nvec_max = 10000, vote_mode = "" ):
    
    if vote_mode == "best":
        nvec, k = find_nvec_k_from_best_count_shape(nsets, nopt_per_set, kmax, nvec_max)
    elif vote_mode == "order":
        nvec, k = find_nvec_k_from_order_count_shape(nsets, nopt_per_set, kmax, nvec_max)
    else:
        raise ValueError(f"Unknown mode {vote_mode}")
    
    return nvec, k

def compress_scores(l):
    
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

    return ivec_to_idx_sorted_compressed, idx_sorted_to_ivec_len_arr
    
def exhaustive_score_to_best_count(l):
    
    ivec_to_idx_sorted_compressed, idx_sorted_to_ivec_len_arr = compress_scores(l)
    
    return exhaustive_score_to_best_count_inner_loop(ivec_to_idx_sorted_compressed, idx_sorted_to_ivec_len_arr)

def exhaustive_score_to_perm_count(l):
    
    ivec_to_idx_sorted_compressed, idx_sorted_to_ivec_len_arr = compress_scores(l)
    
    return exhaustive_score_to_perm_count_inner_loop(ivec_to_idx_sorted_compressed, idx_sorted_to_ivec_len_arr)

def exhaustive_score_to_best_count_brute_force(l):
    
    nvec = len(l)

    prod = 1
    for i in range(nvec):
        prod *= l[i].shape[0]

    if np.iinfo(np.intp).max < prod:
        raise ValueError("Too many observations in vectors")
    
    res = np.zeros(nvec, dtype=np.intp)   
    vals = np.empty(nvec, dtype=np.float64)
    
    ranges = [range(l[i].shape[0]) for i in range(nvec)]
    
    for I in itertools.product(*ranges):
        
        for i in range(nvec):
            vals[i] = l[i][I[i]]
        
        i = np.argmax(vals)
        res[i] += 1
        
    return res

def exhaustive_score_to_perm_count_brute_force(l):
    
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

def exhaustive_score_to_partial_best_count(k, l, best_count = None, opt="opt"):
    
    nvec = len(l)

    if best_count is None:
        ncomb = math.comb(nvec, k)
        best_count = np.zeros((ncomb, k), dtype=np.intp) 
    
    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):
        
        ll = [l[c] for c in comb]

        if opt == "opt":
            best_count[icomb,:] += exhaustive_score_to_best_count(ll)
        elif opt == "brute_force":
            best_count[icomb,:] += exhaustive_score_to_best_count_brute_force(ll)
        else:
            raise ValueError('Unknown backend')
    
    return best_count

def exhaustive_score_to_partial_order_count(k, l, order_count = None, opt="opt"):
    
    nvec = len(l)

    if order_count is None:
        
        nfac = math.factorial(k)
        ncomb = math.comb(nvec, k)

        if np.iinfo(np.intp).max < nfac:
            raise ValueError("Too many vectors")
        
        order_count = np.zeros((ncomb, nfac), dtype=np.intp) 
    
    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):
        
        ll = [l[c] for c in comb]

        if opt == "opt":
            order_count[icomb,:] += exhaustive_score_to_perm_count(ll)
        elif opt == "brute_force":
            order_count[icomb,:] += exhaustive_score_to_perm_count_brute_force(ll)
        else:
            raise ValueError('Unknown backend')
    
    return order_count

def montecarlo_score_to_partial_best_count(k, l, best_count = None, nmc_all = 1000, nrand_max = 10000, cap_nmc = True):
    
    nvec = len(l)
    ncomb = math.comb(nvec, k)
    
    if best_count is None:
        best_count = np.zeros((ncomb, k), dtype=np.intp) 

    if not isinstance(nmc_all, collections.abc.Iterable):
        nmc_per_ncomb, nmc_rem = divmod(nmc_all, ncomb)
        nmc_all = np.full(ncomb, nmc_per_ncomb, dtype = np.intp)
        nmc_all[:nmc_rem] += 1

    if cap_nmc:
        
        for icomb, (comb, nmc) in enumerate(zip(itertools.combinations(range(nvec), k), nmc_all)):
            
            ll = [l[c] for c in comb]

            nobs_tot = 1
            for i in range(nvec):
                nobs_tot *= l[i].shape[0]
                
            if nobs_tot <= nmc :
                best_count[icomb,:] += montecarlo_score_to_best_count(ll)
            else:
                best_count[icomb,:] += montecarlo_score_to_best_count(ll, ll[0][0], nmc = nmc, nrand_max = nrand_max)

    else:
        
        for icomb, (comb, nmc) in enumerate(zip(itertools.combinations(range(nvec), k), nmc_all)):
            ll = [l[c] for c in comb]
            best_count[icomb,:] += montecarlo_score_to_best_count(ll, ll[0][0], nmc = nmc, nrand_max = nrand_max)
            
    return best_count

def montecarlo_score_to_partial_order_count(k, l, order_count = None, nmc_all = 1000, nrand_max = 10000, cap_nmc = True):
    
    nvec = len(l)
    
    nfac = math.factorial(k)
    ncomb = math.comb(nvec, k)

    if order_count is None:
        
        if np.iinfo(np.intp).max < nfac:
            raise ValueError("Too many vectors")
        
        order_count = np.zeros((ncomb, nfac), dtype=np.intp) 

    if not isinstance(nmc_all, collections.abc.Iterable):
        nmc_per_ncomb, nmc_rem = divmod(nmc_all, ncomb)
        nmc_all = np.full(ncomb, nmc_per_ncomb, dtype = np.intp)
        nmc_all[:nmc_rem] += 1

    if cap_nmc:
        
        for icomb, (comb, nmc) in enumerate(zip(itertools.combinations(range(nvec), k), nmc_all)):
            
            ll = [l[c] for c in comb]

            nobs_tot = 1
            for i in range(nvec):
                nobs_tot *= l[i].shape[0]
                
            if nobs_tot <= nmc :
                order_count[icomb,:] += exhaustive_score_to_perm_count(ll)
            else:
                order_count[icomb,:] += montecarlo_score_to_perm_count(ll, ll[0][0], nmc = nmc, nrand_max = nrand_max)

    else:
        
        for icomb, (comb, nmc) in enumerate(zip(itertools.combinations(range(nvec), k), nmc_all)):
            ll = [l[c] for c in comb]
            order_count[icomb,:] += montecarlo_score_to_perm_count(ll, ll[0][0], nmc = nmc, nrand_max = nrand_max)
            
    return order_count

def adaptive_score_to_partial_order_count(k, l, order_count = None, nmc_all = 1000, nmc_step = 10):
    
    nvec = len(l)
    
    if order_count is None:
        
        nfac = math.factorial(k)
        ncomb = math.comb(nvec, k)

        if np.iinfo(np.intp).max < nfac:
            raise ValueError("Too many vectors")
        
        order_count = np.zeros((ncomb, nfac), dtype=np.intp) 
        
    nmc_rem = nmc_all
    
    # I really only need A here ... at least as long as I don't have a matrix-free implementation
    A, p, q = build_sinkhorn_problem(order_count)
    
    while nmc_rem > 0:
        
        nmc = min(nmc_rem, nmc_step)
        
        icomb = get_best_icomb(order_count, A)
        
        comb = unrank_combination(icomb, nvec, k)
        
        ll = [l[c] for c in comb]
        
        order_count[icomb,:] += montecarlo_score_to_perm_count(ll, ll[0][0], nmc = nmc)
        
        nmc_rem -= nmc
        
    return order_count

def get_best_icomb(order_count, A):

    order_count_best = order_count_to_best_count(order_count)
    
    nsets = order_count_best.shape[0]
    nvec = order_count_best.shape[1]
    
    n_tot = order_count_best.sum()
    # reg_eps = 1. / (n_tot + 1)
    reg_eps = 0.
    p, q, dq = build_sinkhorn_rhs(order_count_best, reg_eps = reg_eps)
    
    reg_alpham1 = 0.
    reg_beta = 0.
    
    u, v = sinkhorn_knopp(A, p, q, reg_alpham1 = reg_alpham1, reg_beta = reg_beta)
    M = np.einsum('i,ij,j->ij',u,A,v)
    
    J = build_log_tangent_sinkhorn_problem(M)

    eigvals, eigvects = scipy.linalg.eigh(J)
    
    n_tests_done = order_count_best.sum(axis=1) + 1
    
    dv_norm = np.empty(nsets, dtype=np.float64)
    
    dpq = np.empty((nsets+nvec), dtype=np.float64)
    
    ivecbest = np.argmax(v)
    sf = scipy.special.softmax(v)
    
    for iset in range(nsets):
        
        dpq[:nsets] = 0
        
        dpq[iset] = 1. 
        dpq[nsets:] = dq[iset,:]
        # dpq[nsets:] = 0.
        
        # dloguv , _ , _ , _= np.linalg.lstsq(J,dpq)
        
        # pseudo inversion after symmetric eigendecomposition
        dloguv = np.matmul(eigvects, dpq)
        for i in range(dloguv.shape[0]):
            if abs(eigvals[i]) > 1e-13:
                dloguv[i] /= eigvals[i]
            else:
                dloguv[i] = 0.
        dloguv = np.matmul(eigvects.T, dloguv)
        
        # dv_norm[iset] = np.linalg.norm(v*dloguv[nsets:]) # Computing a norm1 here would be easier (just a dot product)
        dv_norm[iset] = np.dot(v, dloguv[nsets:])
        # dv_norm[iset] = np.linalg.norm(dloguv[nsets:]) 
        # dv_norm[iset] = np.einsum('i,i,i',v,dloguv[nsets:],sf)
        
        dv_norm[iset] /= n_tests_done[iset]

    icomb = np.argmax(dv_norm)
    # icomb = np.argmin(dv_norm)

    return icomb


def score_to_partial_best_count(k, l, best_count = None, method = "exhaustive", nmc_all = 1000, nrand_max = 10000, cap_nmc = True):
    
    nvec = len(l)

    if best_count is None:
        ncomb = math.comb(nvec, k)
        best_count = np.zeros((ncomb, k), dtype=np.intp)   
    
    if method == "exhaustive":
        exhaustive_score_to_partial_best_count(k, l, best_count)
    elif method == "montecarlo":
        montecarlo_score_to_partial_best_count(k, l, best_count, nmc_all = nmc_all, nrand_max = nrand_max, cap_nmc = cap_nmc)
    # elif method == "adaptive":
    #     montecarlo_score_to_partial_best_count(k, l, best_count, nmc_all = 1, nrand_max = 1)
    #     nmc_all = (nmc_all - 1) * math.comb(nvec, k)
    #     adaptive_score_to_partial_best_count(k, l, best_count, nmc_all = nmc_all)
    else:
        raise NotImplementedError
    
    return best_count

def score_to_partial_order_count(k, l, order_count = None, method = "exhaustive", nmc_all = 1000, nrand_max = 10000, cap_nmc = True):
    
    nvec = len(l)

    if order_count is None:
        
        nfac = math.factorial(k)
        ncomb = math.comb(nvec, k)

        if np.iinfo(np.intp).max < nfac:
            raise ValueError("Too many vectors")
        
        order_count = np.zeros((ncomb, nfac), dtype=np.intp)   
    
    if method == "exhaustive":
        exhaustive_score_to_partial_order_count(k, l, order_count)
    elif method == "montecarlo":
        montecarlo_score_to_partial_order_count(k, l, order_count, nmc_all = nmc_all, nrand_max = nrand_max, cap_nmc = cap_nmc)
    elif method == "adaptive":
        montecarlo_score_to_partial_order_count(k, l, order_count, nmc_all = 1, nrand_max = 1)
        nmc_all = (nmc_all - 1) * math.comb(nvec, k)
        adaptive_score_to_partial_order_count(k, l, order_count, nmc_all = nmc_all)
    else:
        raise NotImplementedError
    
    return order_count

def fuse_score_to_partial_vote_count(mode, vote_count, idx_fused):
    
    if mode == "best":
        res = fuse_score_to_partial_best_count(vote_count, idx_fused)
        
    elif mode == "order":
        res = fuse_score_to_partial_order_count(vote_count, idx_fused)
    
    return res

def fuse_score_to_partial_best_count(best_count, idx_fused):
    
    nvec, k = find_nvec_k_from_best_count_shape(best_count.shape[0], best_count.shape[1])
    nfuse = len(idx_fused)
    
    ncomb = best_count.shape[0]
    ncombfuse = math.comb(nfuse, k)

    res = np.zeros((ncombfuse, k), dtype=np.intp)  
    
    ivec_to_idx_fused = np.full(nvec, -1, dtype=np.intp)
    
    for ifuse, idx in enumerate(idx_fused):
        for i in idx:
            ivec_to_idx_fused[i] = ifuse
        
    combfuse_unsorted = np.empty(k, dtype=np.intp)
    combfuse = np.empty(k, dtype=np.intp)

    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):

        for i in range(k):
            combfuse_unsorted[i] = ivec_to_idx_fused[comb[i]]
            
        fuse_perm_sort = np.argsort(combfuse_unsorted)

        if combfuse_unsorted[fuse_perm_sort[0]] < 0:
            continue
        
        for i in range(k):
            combfuse[i] = combfuse_unsorted[fuse_perm_sort[i]]
            
        for i in range(k-1):
            if  combfuse[i+1] == combfuse[i]:
                HasRepeatedEl = True
                break
        else:
            HasRepeatedEl = False
            
        if HasRepeatedEl:
            continue

        icombfuse = rank_combination(combfuse, nfuse, k)
        
        for ibest in range(k):
            
            res[icombfuse, ibest] += best_count[icomb, fuse_perm_sort[ibest]]

    return res

def fuse_score_to_partial_order_count(order_count, idx_fused):
    
    nvec, k = find_nvec_k_from_order_count_shape(order_count.shape[0], order_count.shape[1])
    nfuse = len(idx_fused)
    
    ncomb = order_count.shape[0]
    nfac = order_count.shape[1]
    ncombfuse = math.comb(nfuse, k)

    res = np.zeros((ncombfuse, nfac), dtype=np.intp)  
    
    ivec_to_idx_fused = np.full(nvec, -1, dtype=np.intp)
    
    for ifuse, idx in enumerate(idx_fused):
        for i in idx:
            ivec_to_idx_fused[i] = ifuse
        
    combfuse_unsorted = np.empty(k, dtype=np.intp)
    combfuse = np.empty(k, dtype=np.intp)
    permfuse = np.empty(k, dtype=np.intp)
    invfuse_perm_sort = np.empty(k, dtype=np.intp)

    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):

        for i in range(k):
            combfuse_unsorted[i] = ivec_to_idx_fused[comb[i]]
            
        fuse_perm_sort = np.argsort(combfuse_unsorted)
        for i in range(k):
            invfuse_perm_sort[fuse_perm_sort[i]] = i

        if combfuse_unsorted[fuse_perm_sort[0]] < 0:
            continue
        
        for i in range(k):
            combfuse[i] = combfuse_unsorted[fuse_perm_sort[i]]
            
        for i in range(k-1):
            if  combfuse[i+1] == combfuse[i]:
                HasRepeatedEl = True
                break
        else:
            HasRepeatedEl = False
            
        if HasRepeatedEl:
            continue

        icombfuse = rank_combination(combfuse, nfuse, k)
        
        for iperm in range(nfac):
            
            perm = from_left_lehmer(iperm, k)
            
            for i in range(k):
                permfuse[i] = invfuse_perm_sort[perm[i]]
            
            ipermfuse = left_lehmer(permfuse)

            res[icombfuse, ipermfuse] += order_count[icomb, iperm]

    return res

def order_count_to_best_count_py(order_count, minimize = False):
    
    nsets = order_count.shape[0]
    nopt_per_set = order_count.shape[1]
    nvec, k = find_nvec_k_from_order_count_shape(nsets, nopt_per_set)
    best_count = np.zeros((nsets, k), dtype = order_count.dtype)   
    
    if minimize:
        i_opt = 0
    else:
        i_opt = k-1

    for iset in range(nsets):

        for jperm in range(nopt_per_set):
            
            perm = from_left_lehmer(jperm, k)
            
            best_count[iset, perm[i_opt]] += order_count[iset, jperm]

    return best_count

def order_count_lower_k(order_count, k_new):
    
    kfac_old = order_count.shape[1]
    nvec, k_old = find_nvec_k_from_order_count_shape(order_count.shape[0], order_count.shape[1])

    if k_old < k_new:
        return ValueError(f"k_old should be greater than k_new. Received {k_old = }, {k_new = }.")

    nsets_new = math.comb(nvec, k_new)
    kfac_new = math.factorial(k_new)

    order_count_new = np.zeros((nsets_new, kfac_new), dtype=order_count.dtype)    
    
    comb_new = np.empty(k_new, dtype=np.intp)
    perm_rel = np.empty(k_new, dtype=np.intp)
    perm_old_inv = np.empty(k_old, dtype=np.intp)

    for icomb_old, comb_old in enumerate(itertools.combinations(range(nvec), k_old)):

        for comb_rel in itertools.combinations(range(k_old), k_new):
            
            for i in range(k_new):
                comb_new[i] = comb_old[comb_rel[i]]

            icomb_new = rank_combination(comb_new, nvec, k_new)

            for iperm_old in range(kfac_old):
                
                perm_old = from_left_lehmer(iperm_old, k_old)

                for i in range(k_old):
                    perm_old_inv[perm_old[i]] = i

                for i in range(k_new):
                    perm_rel[i] = perm_old_inv[comb_rel[i]]
                
                perm_new = np.argsort(perm_rel)

                iperm_new = left_lehmer(perm_new)

                order_count_new[icomb_new, iperm_new] += order_count[icomb_old, iperm_old]

    return order_count_new

def condorcet_top_order(order_count, minimize=False):
    
    nvec, k = find_nvec_k_from_order_count_shape(order_count.shape[0], order_count.shape[1])
    
    if (k != 2):
        raise ValueError(f'Expected pairwise comparison data. Received {k}-wise comparison data')
    
    if minimize:
        i_opt = 0
    else:
        i_opt = 1
        
    available_options = list(range(nvec))
    
    res = np.full(nvec, -1, dtype=np.intp)
    
    n_wins = np.zeros(nvec, dtype=np.intp)
    
    for i_round in range(nvec-1):
        
        nrem_vec = nvec-i_round
            
        n_wins[:] = 0
        
        for comb in itertools.combinations(available_options, 2):
            
            iset = comb[1]-1-(comb[0]+3-2*nvec)*comb[0]//2  # Magic formula for unranking binomial coeffs C(n,k) with k==2

            if order_count[iset, i_opt] > order_count[iset, 1-i_opt]:
                n_wins[comb[0]] += 1
            else:
                n_wins[comb[1]] += 1
                
        most_wins = np.argmax(n_wins)
        
        if n_wins[most_wins] == (nrem_vec-1):
            res[nrem_vec-1] = most_wins
            available_options.pop(available_options.index(most_wins))
        else:
            break
    else:
        res[0] = available_options.pop()
        
    return res

# def ranked_pairs_order(order_count, minimize=False):
#     
#     nvec, k = find_nvec_k_from_order_count_shape(order_count)
#     
#     if (k != 2):
#         raise ValueError(f'Expected pairwise comparison data. Received {k}-wise comparison data')
#     
#     if minimize:
#         i_opt = 0
#     else:
#         i_opt = 1
#         
#     available_options = list(range(nvec))
#     
#     res = np.full(nvec, -1, dtype=np.intp)
#     
#     n_wins = np.zeros(nvec, dtype=np.intp)
#     
#     for i_round in range(nvec-1):
#         
#         nrem_vec = nvec-i_round
#             
#         n_wins[:] = 0
#         
#         for comb in itertools.combinations(available_options, 2):
#             
#             iset = comb[1]-1-(comb[0]+3-2*nvec)*comb[0]//2  # Magic formula for unranking binomial coeffs C(n,k) with k==2
# 
#             if order_count[iset, i_opt] > order_count[iset, 1-i_opt]:
#                 n_wins[comb[0]] += 1
#             else:
#                 n_wins[comb[1]] += 1
#                 
#         most_wins = np.argmax(n_wins)
#         
#         if n_wins[most_wins] == (nrem_vec-1):
#             res[nrem_vec-1] = most_wins
#             available_options.pop(available_options.index(most_wins))
#         else:
#             break
#     else:
#         res[0] = available_options.pop()
#         
#     return res

def build_sinkhorn_best_count_mat(nvec, k):
    
    nsets = math.comb(nvec, k)
    
    A = np.zeros((nsets, nvec), dtype=np.float64)
    
    for iset, comb in enumerate(itertools.combinations(range(nvec), k)):

        for j in comb:
            A[iset, j] = 1.
        
    return A

def build_sinkhorn_problem(order_count, reg_eps = 0., minimize=False):
    
    nsets = order_count.shape[0]
    nopt_per_set = order_count.shape[1]

    nvec, k = find_nvec_k_from_order_count_shape(order_count.shape[0], order_count.shape[1])
    
    nopts = nvec
    
    if minimize:
        i_opt = 0
    else:
        i_opt = k-1

    p_int = np.zeros(nsets, dtype=order_count.dtype)    
    q_int = np.zeros(nopts, dtype=order_count.dtype)
    
    total_sum = 0
    
    A = np.zeros((nsets, nopts), dtype=np.float64)
    
    for iset, comb in enumerate(itertools.combinations(range(nvec), k)):
        
        for j in comb:
            A[iset, j] = 1.

        for jperm in range(nopt_per_set):
            
            perm = from_left_lehmer(jperm, k)
            
            val = order_count[iset, jperm]
            
            total_sum += val
            p_int[iset] += val
            q_int[comb[perm[i_opt]]] += val
        
    p = np.empty(nsets, dtype=np.float64)    
    q = np.empty(nopts, dtype=np.float64)
        
    alpha = (1. - reg_eps) / total_sum  
    beta = reg_eps / nsets   
    
    for i in range(nsets):
        p[i] = alpha * p_int[i] + beta
        
    beta = reg_eps / nopts   
        
    for i in range(nopts):
        q[i] = alpha * q_int[i] + beta
                
    return A, p, q

def build_sinkhorn_problem_2(order_count, reg_eps = 0., minimize=False):
    
    nsets = order_count.shape[0]
    nopt_per_set = order_count.shape[1]

    nvec, k = find_nvec_k_from_order_count_shape(order_count.shape[0], order_count.shape[1])
    
    nopts = nvec
    
    if minimize:
        i_opt = 0
    else:
        i_opt = k-1

    p_int = np.zeros(nsets, dtype=order_count.dtype)    
    q_int = np.zeros(nopts, dtype=order_count.dtype)
    
    dq = np.zeros((nsets,nopts), dtype=np.float64)    
    
    total_sum = 0
    
    A = np.zeros((nsets, nopts), dtype=np.float64)
    
    for iset, comb in enumerate(itertools.combinations(range(nvec), k)):
        
        for j in comb:
            A[iset, j] = 1.

        for jperm in range(nopt_per_set):
            
            perm = from_left_lehmer(jperm, k)
            
            val = order_count[iset, jperm]
            
            total_sum += val
            p_int[iset] += val
            q_int[comb[perm[i_opt]]] += val
            
            dq[iset, comb[perm[i_opt]]] += val
        
    p = np.empty(nsets, dtype=np.float64)    
    q = np.empty(nopts, dtype=np.float64)
        
    alpha = (1. - reg_eps) / total_sum  
    beta = reg_eps / nsets   
    
    for i in range(nsets):
        p[i] = alpha * p_int[i] + beta
        
    beta = reg_eps / nopts   
        
    for i in range(nopts):
        q[i] = alpha * q_int[i] + beta
                
    for i in range(nsets):
        dq[i,:] /= dq[i,:].sum()
                
    return A, p, q, dq
            
def build_log_tangent_sinkhorn_problem(M):    
    
    nsets = M.shape[0]
    nopts = M.shape[1]
    
    n = nsets+nopts

    J = np.zeros((n,n), dtype=np.float64)
    
    ml = M.sum(axis=1)
    mr = M.sum(axis=0)
    
    for iset in range(nsets):
        J[iset,iset] = ml[iset]
        
    for iopt in range(nopts):
        i = nsets+iopt
        J[i,i] = mr[iopt]
    
    for iset in range(nsets):
        for iopt in range(nopts):
            i = nsets+iopt

            J[iset,i] = M[iset,iopt]
            J[i,iset] = M[iset,iopt]
                
    return J
