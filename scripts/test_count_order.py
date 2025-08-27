# import os
import math
import numpy as np
import itertools
import pyquickbench

TT = pyquickbench.TimeTrain(names_reduction='avg', include_locs=False)

small_primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271], dtype=np.intp)

# Keywords : Lehmer code, factorial basis
# cf https://github.com/jwodder/permutation

def to_factorial_base(i, n):
    if i == 0:
        return np.zeros(n, dtype=np.intp)
    digits = np.zeros(n, dtype=np.intp)
    j = 1
    while i > 0:
        j += 1
        digits[j-2] = i % j
        i //= j
    return digits

def from_factorial_base(digits):
    n = 0
    base = 1
    for i, d in enumerate(digits, start=1):
        if not (0 <= d <= i):
            raise ValueError(digits)
        n += d * base
        base *= i + 1
    return n
        
def left_lehmer(perm):
    left = list(range(perm.shape[0]-1, -1, -1))
    digits = []
    for x in left[:]:
        i = left.index(perm[x])
        del left[i]
        digits.append(i)
    digits = digits[:-1]
    digits.reverse()
    return from_factorial_base(digits)

def from_left_lehmer(i,n):

    mapping = [0]
    for c in to_factorial_base(i,n):
        for j, y in enumerate(mapping):
            if y >= c:
                mapping[j] += 1
        mapping.append(c)
    mapping = np.array(mapping[:n], dtype=np.intp)
    mapping = n-mapping
    
    return mapping

n = 4
for i in range(math.factorial(n)):
    
    digits_cy = pyquickbench.rankstats.to_factorial_base(i, n)
    digits_py = to_factorial_base(i, n)
    
    assert np.array_equal(digits_cy, digits_py)
    
    i_cy = pyquickbench.rankstats.from_factorial_base(digits_cy)
    i_py = from_factorial_base(digits_py)

    assert i_cy == i_py
    
    perm = from_left_lehmer(i,n)
    assert i == left_lehmer(perm)
    assert i == pyquickbench.rankstats.left_lehmer(np.array(perm))

    

    # print(i, perm)

# exit()

# 
# # @TT.tictoc
# def brute_force_list(*l):
#     
#     nvec = len(l)
#     res = np.zeros(math.factorial(nvec), dtype=np.intp)
#     
#     shapes = [l[i].shape[0] for i in range(nvec)]    
#     ranges = [range(shapes[i]) for i in range(nvec)]
#     
#     vals = np.empty(nvec, dtype=np.float64)
#     
#     for I in itertools.product(*ranges):
#         
#         for i in range(nvec):
#             vals[i] = l[i][I[i]]
#         
#         perm = np.argsort(vals)
#         i = left_lehmer(perm)
#         res[i] += 1
#             
#     return res
# 
# # @TT.tictoc
# def mores_subtle_list_new(*l):
#     
#     nvec = len(l)
#     fac = math.factorial(nvec)
#     res = np.zeros(fac, dtype=np.intp)
#     hsh_ok = np.prod(small_primes[:nvec])
#     
#     shapes = [l[i].shape[0] for i in range(nvec)]
#     
#     nelem_tot = sum(shapes)
#     
#     cum_shapes = np.zeros(nvec, dtype=np.intp)
#     cum_shapes[0] = shapes[0]
#     for i in range(nvec-1):
#         cum_shapes[i+1] = cum_shapes[i] + shapes[i+1]
#     
#     u = np.concatenate(l)
#     idx_sort = np.argsort(u)
#     idx_sorted_to_ivec = np.searchsorted(cum_shapes, idx_sort, side='right')
#     perm = np.empty(nvec, dtype=np.intp)
# 
#     for idx in itertools.combinations(range(nelem_tot), nvec):
#         
#         hsh = 1
#         for i in range(nvec):
#             val = idx_sorted_to_ivec[idx[i]]
#             perm[i] = val
#             hsh *= small_primes[val]
#         
#         if hsh == hsh_ok:
#             i = left_lehmer(perm)
#             res[i] += 1
#             
#     return res
# 
# # @TT.tictoc
# def mores_subtle_list_new_new(*l):
#     
#     nvec = len(l)
#     fac = math.factorial(nvec)
#     res = np.zeros(fac, dtype=np.intp)
#     hsh_ok = np.prod(small_primes[:nvec])
#     
#     shapes = [l[i].shape[0] for i in range(nvec)]
#     
#     nelem_tot = sum(shapes)
#     
#     cum_shapes = np.zeros(nvec, dtype=np.intp)
#     cum_shapes[0] = shapes[0]
#     for i in range(nvec-1):
#         cum_shapes[i+1] = cum_shapes[i] + shapes[i+1]
#     
#     u = np.concatenate(l)
#     idx_sort = np.argsort(u)
#     idx_sorted_to_ivec = np.searchsorted(cum_shapes, idx_sort, side='right')
# 
#     idx_sorted_to_ivec_compressed = []
#     idx_sorted_to_ivec_len = []
#     
#     start = 0
#     end = 0
#     while end < nelem_tot:
# 
#         for end in range(start+1, nelem_tot):
#             if idx_sorted_to_ivec[start] != idx_sorted_to_ivec[end]:
#                 break
#         else:
#             end = nelem_tot
#             
#         idx_sorted_to_ivec_compressed.append(idx_sorted_to_ivec[start])
#         idx_sorted_to_ivec_len.append(end-start)
#         
#         start = end
# 
#     idx_sorted_to_ivec_compressed = np.array(idx_sorted_to_ivec_compressed, dtype=np.intp)
#     idx_sorted_to_ivec_len = np.array(idx_sorted_to_ivec_len, dtype=np.intp)
# 
#     nelem_reduced = idx_sorted_to_ivec_len.shape[0]
#     perm = np.empty(nvec, dtype=np.intp)
# 
#     for idx in itertools.combinations(range(nelem_reduced), nvec):
#         
#         hsh = 1
#         mul = 1
#         for i in range(nvec):
#             val = idx_sorted_to_ivec_compressed[idx[i]]
#             perm[i] = val
#             hsh *= small_primes[val]
#             mul *= idx_sorted_to_ivec_len[idx[i]]
#         
#         if hsh == hsh_ok:
#             i = left_lehmer(perm)
#             res[i] += mul
#             
#     return res
# 
# 
# # @TT.tictoc
# def mores_subtle_list_new_new_new(*l):
#     
# 
#     nvec = len(l)
#     fac = math.factorial(nvec)
#     res = np.zeros(fac, dtype=np.intp)
#     
#     shapes = [l[i].shape[0] for i in range(nvec)]
#     
#     nelem_tot = sum(shapes)
#     
#     cum_shapes = np.zeros(nvec, dtype=np.intp)
#     cum_shapes[0] = shapes[0]
#     for i in range(nvec-1):
#         cum_shapes[i+1] = cum_shapes[i] + shapes[i+1]
#     
#     u = np.concatenate(l)
#     idx_sort = np.argsort(u)
#     idx_sorted_to_ivec = np.searchsorted(cum_shapes, idx_sort, side='right')
# 
#     idx_sorted_to_ivec_compressed = []
#     idx_sorted_to_ivec_len = []
#     
#     start = 0
#     end = 0
#     while end < nelem_tot:
# 
#         for end in range(start+1, nelem_tot):
#             if idx_sorted_to_ivec[start] != idx_sorted_to_ivec[end]:
#                 break
#         else:
#             end = nelem_tot
#             
#         idx_sorted_to_ivec_compressed.append(idx_sorted_to_ivec[start])
#         idx_sorted_to_ivec_len.append(end-start)
#         
#         start = end
# 
#     idx_sorted_to_ivec_compressed = np.array(idx_sorted_to_ivec_compressed, dtype=np.intp)
#     idx_sorted_to_ivec_len = np.array(idx_sorted_to_ivec_len, dtype=np.intp)
#     nelem_reduced = idx_sorted_to_ivec_len.shape[0]
# 
#     ivec_to_idx_sorted_compressed = [[] for i in range(nvec)]
#     
#     for i in range(nelem_reduced):
#         ivec_to_idx_sorted_compressed[idx_sorted_to_ivec_compressed[i]].append(i)
# 
#     for ivec in range(nvec):
#         ivec_to_idx_sorted_compressed[ivec] = np.array(ivec_to_idx_sorted_compressed[ivec])
# 
#     perm = np.empty(nvec, dtype=np.intp)
#     for idx in itertools.product(*ivec_to_idx_sorted_compressed):
# 
#         perm = np.argsort(idx)
# 
#         mul = 1
#         for i in range(nvec):
#             mul *= idx_sorted_to_ivec_len[idx[i]]
# 
#         i = left_lehmer(perm)
#         res[i] += mul
# 
#     return res

    

# n = 20
# m = 20
# p = 20
# q = 20
# 
# v = np.random.random(n)
# w = np.random.random(m)
# y = np.random.random(p)
# z = np.random.random(q)
# 
# print(brute_force_list(v,w,y,z))
# print(mores_subtle_list_new(v,w,y,z))
# print(mores_subtle_list_new_new(v,w,y,z))
# 
# ref = brute_force_list(v,w,y,z)
# assert ref == mores_subtle_list_new(v,w,y,z)
# assert ref == mores_subtle_list_new_new(v,w,y,z)

# nvec = 3
# 
# n = 20
# d = 0.25
# l = [np.random.random(n) + d*ivec for ivec in range(nvec)]

# a = brute_force_list(*l)
# b = mores_subtle_list_new(*l)
# c = mores_subtle_list_new_new(*l)
# d = mores_subtle_list_new_new_new(*l)

# assert np.sum(a) == np.prod(np.array([l[i].shape[0] for i in range(nvec)]))
# assert np.array_equal(a, b)
# assert np.array_equal(a, c)
# assert np.array_equal(a, d)

# d = mores_subtle_list_new_new_new(*l)
# print(d/np.sum(d))

# ref = brute_force_list(v,w)
# assert ref == mores_subtle_list_new(v,w)
# assert ref == mores_subtle_list_new_new(v,w)


# print(brute_force_list(v,w))
# print(mores_subtle_list_new(v,w))
# print(mores_subtle_list_new_new(v,w))


@TT.tictoc
def brute_force_comb_list(k, l):
    
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

def mores_subtle_comb_list_from_fac(k, l):
    
    subtle = pyquickbench.rankstats.score_to_perm_count(l)

    nvec = len(l)
    kfac = math.factorial(k)
    nfac = math.factorial(nvec)
    ncomb = math.comb(nvec, k)
    res = np.zeros((ncomb, kfac), dtype=np.intp)    
    
    nprod = 1
    for ivec in range(nvec):
        nprod *= l[ivec].shape[0]

    sub_perm = np.empty(k, dtype=np.intp)
    perm_out_inv = np.empty(nvec, dtype=np.intp)

    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):
            
        for i in range(nfac):
            
            perm_out = from_left_lehmer(i, nvec)
            
            for ii in range(nvec):
                perm_out_inv[perm_out[ii]] = ii
                
            for ik in range(k):
                sub_perm[ik] = perm_out_inv[comb[ik]]
            
            perm_in = np.argsort(sub_perm)
            j = left_lehmer(perm_in)
            
            res[icomb, j] += subtle[i]
            
        nprod_div = 1
        for ivec in comb:
            nprod_div *= l[ivec].shape[0]
                
        res[icomb, :] //= (nprod // nprod_div)

    return res

@TT.tictoc
def more_subtle_comb_list_latest(k, l):
    
    nvec = len(l)
    nfac = math.factorial(k)
    ncomb = math.comb(nvec, k)

    res = np.zeros((ncomb, nfac), dtype=np.intp)    
    
    for icomb, comb in enumerate(itertools.combinations(range(nvec), k)):
        
        ll = [l[c] for c in comb]
        res[icomb,:] = pyquickbench.rankstats.score_to_perm_count(ll)
        
    return res

nvec = 3

lenlist = [2000] * nvec

score_to_partial_order_count = TT.tictoc(pyquickbench.rankstats.score_to_partial_order_count)

# print(lenlist)

# d = 0.5/2000
d = 1
l = [np.random.random(lenlist[ivec]) + d*ivec for ivec in range(nvec)]

k = 3
# a = brute_force_comb_list(k, l)
# b = more_subtle_comb_list_latest(k, l)
c = score_to_partial_order_count(k, l)
# 
# print(a)
# print(b)
# 
# assert np.array_equal(a, b)

# ntests = 100
# 
# for test in range(ntests):
    # b = more_subtle_comb_list_latest(k, l)


print(c.shape)


print(TT)
