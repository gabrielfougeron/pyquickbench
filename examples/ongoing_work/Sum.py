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

small = 1e-40

def builtin_sum(x):
    return sum(x)

def naive_sum(x):
    s = 0.
    for i in range(x.shape[0]):
        s += x[i]
    return s    
    
nb_naive_sum = nb.jit(naive_sum, **numba_opt_dict)
nb_naive_sum.__name__ = "nb_naive_sum"

def np_sum(x):
    return np.sum(x)

def m_fsum(x):
    return m.fsum(x)

@nb.jit(**numba_opt_dict)
def taylor_poly(n, alpha):
    
    x = np.zeros((n),dtype=np.float64)
    cur_term = 1.
    x[0] = cur_term

    for i in range(1, x.shape[0]):

        cur_term = cur_term * (alpha / i)
        x[i] = cur_term

    return x

def setup(alpha, n):
    return {'x': taylor_poly(n, -alpha)}

@functools.cache
def exact_sum(alpha, n):
    y = setup(alpha, n)['x']
    return m.fsum(y)
    
def compute_error_relative_to_exp(f, x):
    
    ex_res =  m.exp(-x[1])
    res = f(x)
    rel_err = abs(ex_res - res) / abs(ex_res)
    
    return rel_err + small

def rel_upper_bound_lagrange(alpha, n):
    
    return small + abs(alpha)**(n+1) / scipy.special.factorial(n+1)

def compute_error_relative_to_fsum(f, x):
    
    ex_res =  exact_sum(-x[1], x.shape[0])
    res = f(x)
    rel_err = abs(ex_res - res) / abs(ex_res)
    
    return rel_err + 1e-40

# sphinx_gallery_start_ignore

basename = 'sum_bench_accuracy'
error_filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

all_args = {
    "alpha": np.array([float(alpha) for alpha in range(500)]),
    # "n" : np.array([2**n for n in range(2,20)]),
    "n" : np.array([2**n for n in range(2,20)]),
}

all_funs = [
    # naive_sum   ,
    nb_naive_sum,
    # builtin_sum ,
    # np_sum      ,
    # m_fsum      ,
]

# %%

all_error_funs = { f.__name__ :  functools.partial(compute_error_relative_to_fsum, f) for f in all_funs if f is not m_fsum}

all_errors = pyquickbench.run_benchmark(
    all_args                        ,
    all_error_funs                  ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    # filename = error_filename       ,
#     show = True                             ,
    StopOnExcept = True,
    ShowProgress = True,
    nproc = 8   ,
    pooltype = "phony"   ,
    # pooltype = "thread"   ,
    # pooltype = "process"   ,
)

plot_intent = {
    "alpha" : 'points' ,
    "n" : 'subplot_grid_y'  ,
}

pyquickbench.plot_benchmark(
    all_errors      ,
    all_args        ,
    all_error_funs  ,
    plot_intent = plot_intent,
    title = "Relative error for increasing conditionning"   ,
    ylabel = "error"    ,
    show = True
)

# %%
#



# %%
# 
# def prepare_x(n):
#     x = np.random.random(n)
#     return {'x': x}
# 
# basename = 'sum_bench_time'
# timings_filename = os.path.join(timings_folder,basename+'.npy')
# 
# all_sizes = np.array([2**n for n in range(20)])
# 
# all_funs = [
#     naive_sum,
#     nb_naive_sum,
#     builtin_sum,
#     np_sum,
#     m_fsum,
# ]
# 
# pyquickbench.run_benchmark(
#     all_sizes                       ,
#     all_funs                        ,
#     setup = prepare_x               ,
#     filename = timings_filename     ,
#     title = "Time (s) as a function of array size"   ,
#     show = True                             ,
# )

