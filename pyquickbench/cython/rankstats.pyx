
import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.stdlib cimport malloc, free, rand
from libc.string cimport memset

import itertools
import math

cpdef void AssertFalse():
    assert False

@cython.cdivision(True)
cdef inline Py_ssize_t _binomial(Py_ssize_t n, Py_ssize_t k) noexcept nogil:

    cdef Py_ssize_t i, res

    res = 1

    for i in range(k):
        res *= n-i
        res //= i+1

    return res

@cython.cdivision(True)
cdef inline void _to_factorial_base(Py_ssize_t i, Py_ssize_t n, Py_ssize_t *out) noexcept nogil:

    memset(out, 0, sizeof(Py_ssize_t)*n)

    cdef Py_ssize_t j 

    j = 1
    while i > 0:
        j += 1
        out[j-2] = i % j
        i //= j

@cython.cdivision(True)
cdef inline Py_ssize_t _from_factorial_base(Py_ssize_t n, Py_ssize_t *digits) noexcept nogil:

    cdef Py_ssize_t res = 0
    cdef Py_ssize_t base = 1
    cdef Py_ssize_t i

    for i in range(n):

        res += digits[i] * base
        base *= i + 2

    return res

def to_factorial_base(Py_ssize_t i, Py_ssize_t n):

    cdef np.ndarray[Py_ssize_t, ndim=1, mode='c'] res = np.empty((n), dtype=np.intp)
    
    _to_factorial_base(i, n, &res[0])
    
    return res

def from_factorial_base(Py_ssize_t[::1] digits):
    return _from_factorial_base(digits.shape[0], &digits[0])

@cython.cdivision(True)
cdef inline void _count_inversions(Py_ssize_t n, Py_ssize_t *perm, Py_ssize_t *out) noexcept nogil:

    cdef Py_ssize_t i, j
    cdef Py_ssize_t invcnt

    for i in range(1,n):
        invcnt = 0
        for j in range(i):
            if perm[j] > perm[i]:
                invcnt += 1
        out[i-1] = invcnt

@cython.cdivision(True)
cdef inline void _from_inversions(Py_ssize_t i, Py_ssize_t m, Py_ssize_t * digits, Py_ssize_t *perm) noexcept nogil:

    cdef Py_ssize_t k, j, d

    perm[0] = 0
    for k in range(m):
        for j in range(k+1):
            if perm[j] >= digits[k]:
                perm[j] += 1
        perm[k+1] = digits[k]
        
    for k in range(m):
        perm[k] = m - perm[k]
    perm[m] = m - perm[m]

@cython.cdivision(True)
cdef inline Py_ssize_t _left_lehmer(Py_ssize_t n, Py_ssize_t *perm, Py_ssize_t *digits) noexcept nogil:
    _count_inversions(n, &perm[0], digits)
    return _from_factorial_base(n-1, digits)

cpdef Py_ssize_t left_lehmer(Py_ssize_t[::1] perm):

    cdef Py_ssize_t n = perm.shape[0]
    cdef Py_ssize_t *dd = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*(n-1))

    cdef Py_ssize_t res = _left_lehmer(n, &perm[0], dd)

    free(dd)

    return res

cdef inline void _from_left_lehmer(Py_ssize_t i, Py_ssize_t n, Py_ssize_t * digits, Py_ssize_t *perm) noexcept nogil:

    cdef Py_ssize_t m = n-1

    _to_factorial_base(i, m, digits)
    _from_inversions(i, m, digits, perm)

def from_left_lehmer(Py_ssize_t i, Py_ssize_t n):

    cdef Py_ssize_t *digits = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*(n-1))
    cdef np.ndarray[Py_ssize_t, ndim=1, mode='c'] res = np.empty((n), dtype=np.intp)

    _from_left_lehmer(i, n, digits, &res[0])

    free(digits)

    return res

cdef inline Py_ssize_t _rank_combination(Py_ssize_t[::1] comb, Py_ssize_t n, Py_ssize_t k) noexcept nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t res = _binomial(n, k) - 1

    for i in range(k):
        res -= _binomial(n-1-comb[i], k-i)

    return res

def rank_combination(Py_ssize_t[::1] comb, Py_ssize_t n, Py_ssize_t k):
    return _rank_combination(comb, n, k)

cdef inline void _unrank_combination(Py_ssize_t r, Py_ssize_t n, Py_ssize_t k, Py_ssize_t *res) noexcept nogil:

    cdef Py_ssize_t restRank = _binomial(n, k) - 1 - r
    cdef Py_ssize_t onesLeft = k
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t nm1 = n-1
    cdef Py_ssize_t binoVal

    while onesLeft > 0:

        binoVal = _binomial(nm1-j, onesLeft)

        if restRank >= binoVal:

            res[k-onesLeft] = j
            onesLeft -= 1
            restRank -= binoVal

        j += 1

def unrank_combination(Py_ssize_t r, Py_ssize_t n, Py_ssize_t k):

    cdef np.ndarray[Py_ssize_t, ndim=1, mode='c'] res = np.empty(k, dtype=np.intp)

    _unrank_combination(r, n, k, &res[0])

    return res

ctypedef fused num_t:
    Py_ssize_t
    float
    double

cdef inline void _insertion_argsort(Py_ssize_t n, num_t* arr, Py_ssize_t* perm) noexcept nogil:

    cdef Py_ssize_t i, j
    cdef num_t key

    perm[0] = 0

    for i in range(1, n):
    
        key = arr[i]
        j = i-1

        while (j >= 0  and  arr[perm[j]] > key):

            perm[j + 1] = perm[j]
            j -= 1

        perm[j + 1] = i

def insertion_argsort(num_t[::1] arr):

    cdef np.ndarray[Py_ssize_t, ndim=1, mode='c'] res = np.empty(arr.shape[0], dtype=np.intp)
    
    _insertion_argsort(arr.shape[0], &arr[0], &res[0])

    return res

def exhaustive_score_to_perm_count_inner_loop(list l, Py_ssize_t[::1] cnt):

    cdef Py_ssize_t nvec = len(l)
    cdef Py_ssize_t fac = math.factorial(nvec)

    cdef Py_ssize_t[::1] res = np.zeros(fac, dtype=np.intp)

    cdef Py_ssize_t[::1] tmp
    cdef Py_ssize_t ivec, it, val
    cdef Py_ssize_t mul

    cdef Py_ssize_t **ivec_to_idx_ptr = <Py_ssize_t**> malloc(sizeof(Py_ssize_t*)*nvec)
    
    cdef Py_ssize_t *ranges = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*nvec)
    cdef Py_ssize_t *itr = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*nvec)
    memset(itr, 0, sizeof(Py_ssize_t)*nvec)

    cdef Py_ssize_t nmax = 1
    for ivec in range(nvec):
        tmp = l[ivec]
        ranges[ivec] = tmp.shape[0]
        nmax *= ranges[ivec]
        ivec_to_idx_ptr[ivec] = &tmp[0]

    cdef Py_ssize_t *dd = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*(nvec-1))
    cdef Py_ssize_t *perm = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*nvec)
    cdef Py_ssize_t *idx = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*nvec)

    for it in range(nmax):

        mul = 1
        for ivec in range(nvec):
            val = ivec_to_idx_ptr[ivec][itr[ivec]]
            idx[ivec] = val
            mul *= cnt[val]

        _insertion_argsort(nvec, idx, perm)

        i = _left_lehmer(nvec, perm, dd)
        res[i] += mul

        for ivec in range(nvec):
            itr[ivec] += 1
            if itr[ivec] != ranges[ivec]:
                break
            else:
                itr[ivec] = 0

    free(ivec_to_idx_ptr)
    free(dd)
    free(perm)
    free(idx)
    free(ranges)
    free(itr)
    
    return np.asarray(res)

def montecarlo_score_to_perm_count(list l, num_t key, Py_ssize_t nmc = 1000, Py_ssize_t nrand_max = 10000):

    cdef Py_ssize_t nvec = len(l)
    cdef Py_ssize_t fac = math.factorial(nvec)

    cdef Py_ssize_t[::1] res = np.zeros(fac, dtype=np.intp)

    cdef Py_ssize_t[::1] tmp_intp
    cdef Py_ssize_t ivec, it, val

    cdef Py_ssize_t **idx_all = <Py_ssize_t**> malloc(sizeof(Py_ssize_t*)*nvec)
    
    cdef num_t **ivec_to_idx_ptr = <num_t**> malloc(sizeof(num_t*)*nvec)
    cdef num_t[::1] tmp
    for ivec in range(nvec):
        tmp = l[ivec]
        ivec_to_idx_ptr[ivec] = &tmp[0]

    cdef Py_ssize_t *dd = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*(nvec-1))
    cdef Py_ssize_t *perm = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*nvec)
    cdef num_t *idx = <num_t*> malloc(sizeof(num_t)*nvec)

    cdef Py_ssize_t nit
    cdef Py_ssize_t nmc_rem = nmc

    while nmc_rem > 0 :

        nit = min(nrand_max, nmc_rem)
        nmc_rem -= nit

        idx_all_list = []
        for ivec in range(nvec):
            tmp_intp = np.random.randint(l[ivec].shape[0], size = nit, dtype=np.intp)
            idx_all_list.append(tmp_intp)
            idx_all[ivec] = &tmp_intp[0]

        for it in range(nit):

            for ivec in range(nvec):
                idx[ivec] = ivec_to_idx_ptr[ivec][idx_all[ivec][it]]

            _insertion_argsort(nvec, idx, perm)

            i = _left_lehmer(nvec, perm, dd)
            res[i] += 1

    free(idx_all)
    free(ivec_to_idx_ptr)
    free(dd)
    free(perm)
    free(idx)
    
    return np.asarray(res)

@cython.cdivision(True)
cdef inline Py_ssize_t _KendallTauInversions(Py_ssize_t *perm_a, Py_ssize_t *perm_b, Py_ssize_t n) noexcept nogil:

    cdef Py_ssize_t i, j
    cdef Py_ssize_t res = 0

    for j in range(1,n):
        for i in range(j):
            if (perm_a[i] < perm_a[j]) != (perm_b[i] < perm_b[j]):
                res += 1

    return res

cpdef double KendallTauDistance(Py_ssize_t[::1] perm_a, Py_ssize_t[::1] perm_b):

    cdef Py_ssize_t n = perm_a.shape[0]

    if perm_b.shape[0] != n:
        raise ValueError(f"The two permutations should have the same size, but they do not. {perm_a.shape[0] = } {perm_b.shape[0] = }")

    cdef Py_ssize_t ninv = _KendallTauInversions(&perm_a[0], &perm_b[0], n)
    cdef double dist = (2.*ninv) / (n*(n-1))

    return dist

cpdef double KendallTauRankCorrelation(Py_ssize_t[::1] perm_a, Py_ssize_t[::1] perm_b):
    return 1. - 2*KendallTauDistance(perm_a, perm_b)

cpdef (Py_ssize_t, Py_ssize_t) find_nvec_k_from_order_count_shape(Py_ssize_t[:,::1] order_count, Py_ssize_t kmax = 100, Py_ssize_t nvec_max = 100):
    
    # Find nvec, k such that factorial(k) == order_count.shape[1] and comb(k,nvec) == order_count.shape[0]
    
    cdef Py_ssize_t nsets = order_count.shape[0]
    cdef Py_ssize_t  nopt_per_set = order_count.shape[1]
    
    cdef Py_ssize_t nvec, k
    cdef Py_ssize_t j

    cdef Py_ssize_t kfac = 1
    for k in range(1,kmax):
        kfac *= k
        if kfac == nopt_per_set:
            break
    else:
        raise ValueError("Could not determine k")
    
    cdef Py_ssize_t *c_arr = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*nvec_max)
    for nvec in range(nvec_max):
        c_arr[nvec] = 1

    for nvec in range(1,k):
        for j in range(nvec-1,0,-1):
            c_arr[j] += c_arr[j-1] 

    for nvec in range(k,nvec_max):
        for j in range(nvec-1,0,-1):
            c_arr[j] += c_arr[j-1] 

        if c_arr[k] == nsets:
            break
    else:
        raise ValueError("Could not determine nvec")
    
    free(c_arr)

    return nvec, k

cdef void _project_order_count_best(Py_ssize_t[:,::1] order_count, Py_ssize_t[:,::1] order_count_best, Py_ssize_t k, bint minimize=False) noexcept nogil:
    
    cdef Py_ssize_t nsets = order_count.shape[0]
    cdef Py_ssize_t nopt_per_set = order_count.shape[1]

    cdef Py_ssize_t nvec = order_count_best.shape[1]

    cdef Py_ssize_t i_opt

    if minimize:
        i_opt = 0
    else:
        i_opt = k-1

    memset(&order_count_best[0,0], 0, sizeof(Py_ssize_t)*nsets*nvec)

    cdef Py_ssize_t *digits = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*(k-1))
    cdef Py_ssize_t *perm = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*k)
    cdef Py_ssize_t *comb = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*k)

    for iset in range(nsets):

        _unrank_combination(iset, nvec, k, comb)

        for jperm in range(nopt_per_set):
            
            _from_left_lehmer(jperm, k, digits, perm)

            order_count_best[iset, comb[perm[i_opt]]] += order_count[iset, jperm]

    free(digits)
    free(perm)
    free(comb)

def project_order_count_best(Py_ssize_t[:,::1] order_count, bint minimize=False):
    
    cdef Py_ssize_t nsets = order_count.shape[0]
    cdef Py_ssize_t nvec, k
    nvec, k = find_nvec_k_from_order_count_shape(order_count)
    cdef Py_ssize_t[:,::1] order_count_best = np.empty((nsets, nvec), dtype=np.intp)   

    _project_order_count_best(order_count, order_count_best, k, minimize)

    return np.asarray(order_count_best)
    
@cython.cdivision(True)
cpdef void _build_sinkhorn_rhs(
    Py_ssize_t[:,::1] order_count_best,
    double[::1] p       ,
    double[::1] q       ,
    double[:,::1] dq    ,
    double reg_eps = 0. ,
) noexcept nogil:
    
    cdef Py_ssize_t nsets = order_count_best.shape[0]
    cdef Py_ssize_t nvec = order_count_best.shape[1]

    cdef Py_ssize_t iset, ivec
    cdef double val

    memset(&p[0], 0, sizeof(double)*nsets)
    memset(&q[0], 0, sizeof(double)*nvec)
    
    for iset in range(nsets):
        for ivec in range(nvec):
            
            val = order_count_best[iset, ivec]
            
            p[iset] += val
            q[ivec] += val
            dq[iset, ivec] = val
    
    cdef double total_sum = 0
            
    for iset in range(nsets):
        val = p[iset]
        total_sum += val
        for ivec in range(nvec):
            dq[iset, ivec] /= val

    cdef double alpha, beta
        
    alpha = (1. - reg_eps) / total_sum  
    beta = reg_eps / nsets   
    
    for i in range(nsets):
        p[i] = alpha * p[i] + beta
        
    beta = reg_eps / nvec   
        
    for i in range(nvec):
        q[i] = alpha * q[i] + beta

def build_sinkhorn_rhs(
    Py_ssize_t[:,::1] order_count_best  ,
    double reg_eps = 0.                 ,
):
    
    nsets = order_count_best.shape[0]
    nvec = order_count_best.shape[1]

    cdef double[::1] p = np.empty(nsets, dtype=np.float64)    
    cdef double[::1] q = np.empty(nvec, dtype=np.float64)       
    cdef double[:,::1] dq = np.empty((nsets, nvec), dtype=np.float64)

    _build_sinkhorn_rhs(order_count_best, p, q, dq, reg_eps)
    
    return np.asarray(p), np.asarray(q), np.asarray(dq)

# @cython.cdivision(True)
# def adaptive_find_best_icomb_update(
#     Py_ssize_t[:,::1] order_count       ,
#     Py_ssize_t[:,::1] order_count_best  ,
#     Py_ssize_t ntot                     ,
#     double[::1] p                       ,
#     double[::1] q                       ,
#     double[:,::1] dq                    ,
# ):
# 
#     cdef Py_ssize_t nsets = order_count.shape[0]
#     cdef Py_ssize_t nvec, k
#     nvec, k = find_nvec_k_from_order_count_shape(order_count)
# 
#     _project_order_count_best(order_count, order_count_best, k, False)
# 
#     cdef double reg_eps = 1. / (n_tot + 1)
# 
#     _build_sinkhorn_rhs(order_count_best, p, q, dq, reg_eps)
# 
#         
#     cdef double reg_alpham1 = 0.
#     cdef double reg_beta = 0.