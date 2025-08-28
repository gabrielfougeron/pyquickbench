
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

def to_factorial_base(Py_ssize_t i, Py_ssize_t n):

    cdef np.ndarray[Py_ssize_t, ndim=1, mode='c'] res = np.empty((n), dtype=np.intp)
    
    _to_factorial_base(i, n, &res[0])
    
    return res

def from_factorial_base(Py_ssize_t[::1] digits):
    return _from_factorial_base(digits.shape[0], &digits[0])

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

    _to_factorial_base(i, n-1, digits)

    cdef Py_ssize_t k, j, d, m

    m = n-1

    perm[0] = 0
    for k in range(m):
        for j in range(k+1):
            if perm[j] >= digits[k]:
                perm[j] += 1
        perm[k+1] = digits[k]
        
    for k in range(n):
        perm[k] = m - perm[k]

def from_left_lehmer(Py_ssize_t i, Py_ssize_t n):

    cdef Py_ssize_t *digits = <Py_ssize_t*> malloc(sizeof(Py_ssize_t)*(n-1))
    cdef np.ndarray[Py_ssize_t, ndim=1, mode='c'] res = np.empty((n), dtype=np.intp)

    _from_left_lehmer(i, n, digits, &res[0])

    free(digits)

    return res

ctypedef fused num_t:
    Py_ssize_t
    float
    double

cdef void insertion_argsort(Py_ssize_t n, num_t* arr, Py_ssize_t* perm) noexcept nogil:

    cdef Py_ssize_t i, j, k
    cdef num_t key

    for i in range(n):
        perm[i] = i

    for i in range(1, n):
        k = perm[i]
        key = arr[k]
        j = i-1

        while (j >= 0  and  arr[perm[j]] > key):

            perm[j + 1] = perm[j]
            j -= 1

        perm[j + 1] = k

def score_to_perm_count_inner_loop(list l, Py_ssize_t[::1] cnt):

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

        insertion_argsort(nvec, idx, perm)

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
