"""
Benchmark of Error-Free Transforms for summation
================================================
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

# ForceBenchmark = True
ForceBenchmark = False

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)


def builtin_sum(x):
    return sum(x)

def naive_sum(x):
    s = 0.
    for i in range(x.shape[0]):
        s += x[i]
    return s    
    
nb_naive_sum = nb.jit(naive_sum, **numba_opt_dict)

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



@functools.cache
def exact_sum(alpha):
    y = setup(alpha)
    return m.fsum(y)
    
def compute_error(f, x):
    
    ex_res =  exact_sum(-x[1])
    res = f(x)
    rel_err = abs(ex_res - res) / abs(ex_res)
    
    return rel_err + 1e-40


dpi = 150

figsize = (1600/dpi, 800 / dpi)

fig, axs = plt.subplots(
    nrows = 1,
    ncols = 1,
    sharex = True,
    sharey = True,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = True,
)

basename = 'sum_bench_accuracy'
error_filename = os.path.join(timings_folder,basename+'.npy')

# all_alphas = np.array([float(2**i) for i in range(10)])
all_alphas = np.array([float(alpha) for alpha in range(500)])

all_funs = [
    naive_sum,
    builtin_sum,
    np_sum,
    m_fsum,
    SumK_1,
    SumK_2,
    SumK_3,
    FastSumK_1,
    FastSumK_2,
    FastSumK_3,
]

all_error_funs = { f.__name__ :  functools.partial(compute_error, f) for f in all_funs if f is not m_fsum}

all_times = pyquickbench.run_benchmark(
    all_alphas                      ,
    all_error_funs                  ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    filename = error_filename       ,
    ForceBenchmark = ForceBenchmark ,
)

pyquickbench.plot_benchmark(
    all_times                               ,
    all_alphas                              ,
    all_error_funs                          ,
    fig = fig                               ,
    ax = axs                                ,
    title = "Relative error for increasing conditionning"   ,
)
    
plt.tight_layout()

plt.show()

# sphinx_gallery_end_ignore


# %%

def prepare_x(n):
    x = np.random.random(n)
    return [(x, 'x')]

# sphinx_gallery_start_ignore
dpi = 150

figsize = (1600/dpi, 800 / dpi)

fig, axs = plt.subplots(
    nrows = 1,
    ncols = 1,
    sharex = True,
    sharey = True,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = True,
)


basename = 'sum_bench_time'
timings_filename = os.path.join(timings_folder,basename+'.npy')

all_sizes = np.array([2**n for n in range(21)])

all_funs = [
    naive_sum,
    builtin_sum,
    np_sum,
    m_fsum,
    SumK_1,
    SumK_2,
    SumK_3,
    FastSumK_1,
    FastSumK_2,
    FastSumK_3,
]

all_times = pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    setup = prepare_x               ,
    filename = timings_filename     ,
    ForceBenchmark = ForceBenchmark ,
)

pyquickbench.plot_benchmark(
    all_times                               ,
    all_sizes                               ,
    all_funs                                ,
    fig = fig                               ,
    ax = axs                                ,
    title = "Time (s) as a function of array size"   ,
)
    
plt.tight_layout()

plt.show()

# sphinx_gallery_end_ignore
