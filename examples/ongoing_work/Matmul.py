"""
Summing elements of a numpy array
=================================
"""

# %% 
# This benchmark compares accuracy and efficiency of several summation algorithms in floating point arithmetics

# sphinx_gallery_start_ignore

import os
import sys

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import functools
import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy
import numba as nb

numba_opt_dict = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

def Three_loops_python(a, b, c):

    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] += a[i,k]*b[k,j]

Three_loops_numba_serial = nb.jit(Three_loops_python,**numba_opt_dict)
Three_loops_numba_auto_parallel = nb.jit(Three_loops_python,parallel=True,**numba_opt_dict)

@nb.jit(parallel=True,**numba_opt_dict)
def Three_loops_numba_parallel(a, b, c):

    for i in nb.prange(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] += a[i,k]*b[k,j]

@nb.jit(parallel=True,**numba_opt_dict)
def Three_loops_numba_parallel_noreduce(a, b, c):

    for i in nb.prange(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] = c[i,j] + a[i,k]*b[k,j]

def numpy_matmul(a, b, c):
    np.matmul(a, b, out=c)

def setup_abc(P, R, Q, real_dtype):
    a = np.random.random((P,R)).astype(dtype=real_dtype)
    b = np.random.random((R,Q)).astype(dtype=real_dtype)
    c = np.zeros((P,Q),dtype=real_dtype)
    return {'a':a, 'b':b, 'c':c}

# sphinx_gallery_start_ignore

basename = 'matmul_timings'
filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

all_args = {
    "P" : [32 * (2 ** k) for k in range(2)]     ,
    "Q" : [32 * (2 ** k) for k in range(2)]     ,
    "R" : [32 * (2 ** k) for k in range(2)]     ,
    "real_dtype": [np.float32, np.float64]     ,
}

all_funs = [
    Three_loops_python                  ,
    Three_loops_numba_serial            ,
    Three_loops_numba_auto_parallel     ,
    Three_loops_numba_parallel_noreduce ,
    numpy_matmul                        ,
]

# %%

all_errors = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup_abc       ,
    filename = filename     ,
    StopOnExcept = True     ,
    ShowProgress = True     ,
)

# plot_intent = {
#     "alpha" : 'points' ,
#     "n" : 'subplot_grid_y'  ,
# }
# 
# pyquickbench.plot_benchmark(
#     all_errors      ,
#     all_args        ,
#     all_error_funs  ,
#     plot_intent = plot_intent,
#     title = "Relative error for increasing conditionning"   ,
#     ylabel = "error"    ,
#     show = True
# )