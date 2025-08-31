
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
    # reg                 ,
    Py_ssize_t numItermax = 1000    ,
    Py_ssize_t check_err_every = 100,
    double stopThr = 1e-9           ,
    warmstart = None                ,
):

    r"""


    cf POT

    @article{flamary2021pot,
    author  = {R{\'e}mi Flamary and Nicolas Courty and Alexandre Gramfort and Mokhtar Z. Alaya and Aur{\'e}lie Boisbunon and Stanislas Chambon and Laetitia Chapel and Adrien Corenflos and Kilian Fatras and Nemo Fournier and L{\'e}o Gautheron and Nathalie T.H. Gayraud and Hicham Janati and Alain Rakotomamonjy and Ievgen Redko and Antoine Rolet and Antony Schutz and Vivien Seguy and Danica J. Sutherland and Romain Tavenard and Alexander Tong and Titouan Vayer},
    title   = {POT: Python Optimal Transport},
    journal = {Journal of Machine Learning Research},
    year    = {2021},
    volume  = {22},
    number  = {78},
    pages   = {1-8},
    url     = {http://jmlr.org/papers/v22/20-451.html}
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

    cdef Py_ssize_t i_iter
    cdef Py_ssize_t  i_iter_rem = check_err_every-1
    cdef Py_ssize_t i

    cdef double tmp

    cdef double[::1] uM = np.empty((dim_b), dtype=np.float64)
    cdef double[::1] tmp_arr= np.empty((dim_b), dtype=np.float64)

    cdef double err = 1.
    for i_iter in range(numItermax):


        scipy.linalg.cython_blas.dgemv(transn,&dim_b,&dim_a,&one_double,&M[0,0],&dim_b,&u[0],&int_one,&zero_double,&uM[0],&int_one)

        if i_iter % check_err_every == i_iter_rem:

            for i in range(dim_b):
                tmp_arr[i] = v[i] * uM[i]

            scipy.linalg.cython_blas.daxpy(&dim_b, &minusone_double, &b[0], &int_one, &tmp_arr[0], &int_one)
            err = csqrt(scipy.linalg.cython_blas.ddot(&dim_b, &tmp_arr[0], &int_one, &tmp_arr[0], &int_one))


            print(f'{i_iter = }')
            print(err)

            if err < stopThr:
                break

        for i in range(dim_b):
            v[i] = b[i] / uM[i]

        scipy.linalg.cython_blas.dgemv(transt,&dim_b,&dim_a,&one_double,&M[0,0],&dim_b,&v[0],&int_one,&zero_double,&u[0],&int_one)
        for i in range(dim_a):
            u[i] = a[i] / u[i]

    return (np.asarray(u), np.asarray(v))