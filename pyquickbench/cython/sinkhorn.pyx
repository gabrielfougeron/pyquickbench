
import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport sqrt as csqrt
from libc.stdlib cimport malloc, free, rand
from libc.string cimport memset

cimport scipy.linalg.cython_blas

cdef char *transn = 'n'
cdef char *transt = 't'
cdef int int_zero = 0
cdef int int_one = 1
cdef double zero_double = 0.
cdef double one_double = 1.
cdef double minusone_double = -1.

@cython.cdivision(True)
def sinkhorn_knopp(
    double[::1] a                   ,
    double[::1] b                   ,
    double[:,::1] M                 ,
    double reg_alpham1 = 0.         ,
    double reg_beta = 0.            ,
    Py_ssize_t numItermax = 1000    ,
    Py_ssize_t check_err_every = 100,
    double stopThr = 1e-9           ,
    warmstart = None                ,
):

    r"""

    Regularized Sinkhorn-Knopp algorithm with Gamma(alpha, beta) prior.

    cf

    @article{qu2025sinkhorn,
    title={On sinkhorn{\'}s algorithm and choice modeling},
    author={Qu, Zhaonan and Galichon, Alfred and Gao, Wenzhi and Ugander, Johan},
    journal={Operations Research},
    year={2025},
    publisher={INFORMS}
    }

    """

    cdef int dim_a = a.shape[0]
    cdef int dim_b = b.shape[0]

    assert dim_a == M.shape[0]
    assert dim_b == M.shape[1]

    cdef double[::1] u, v

    if warmstart is None:
        u = np.full(dim_a, 1./dim_a ,dtype=np.float64)
        v = np.full(dim_b, 1./dim_b ,dtype=np.float64)
    else:
        u = warmstart[0]
        v = warmstart[1]

    cdef double *uptr = &u[0]
    cdef double *vptr = &v[0]

    cdef double *aptr = &a[0]
    cdef double *bptr = &b[0]

    cdef Py_ssize_t i_iter
    cdef Py_ssize_t  i_iter_rem = check_err_every-1
    cdef Py_ssize_t i

    cdef double tmp
    cdef double * uM = <double*> malloc(sizeof(double)*dim_b)

    cdef double err = 1.
    for i_iter in range(numItermax):

        scipy.linalg.cython_blas.dgemv(transn,&dim_b,&dim_a,&one_double,&M[0,0],&dim_b,uptr,&int_one,&zero_double,uM,&int_one)

        if i_iter % check_err_every == i_iter_rem:

            err = 0.
            for i in range(dim_b):
                tmp = vptr[i] * (uM[i] + reg_beta) - (bptr[i] + reg_alpham1)
                err += tmp * tmp
            err = csqrt(err)

            if err < stopThr:
                break

        for i in range(dim_b):
            vptr[i] = (bptr[i] + reg_alpham1) / (uM[i] + reg_beta)

        scipy.linalg.cython_blas.dgemv(transt,&dim_b,&dim_a,&one_double,&M[0,0],&dim_b,vptr,&int_one,&zero_double,uptr,&int_one)
        for i in range(dim_a):
            uptr[i] = aptr[i] / uptr[i]

    free(uM)

    return (np.asarray(u), np.asarray(v))