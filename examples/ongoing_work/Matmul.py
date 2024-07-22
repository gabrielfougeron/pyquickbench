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

import matplotlib.pyplot as plt
import numpy as np
import math as m
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

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

def python(a, b, c):

    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] += a[i,k]*b[k,j]

numba_serial = nb.jit(python,**numba_opt_dict)
numba_serial.__name__ = "numba_serial"
numba_auto_parallel = nb.jit(python,parallel=True,**numba_opt_dict)
numba_auto_parallel.__name__ = "numba_auto_parallel"

@nb.jit(parallel=True,**numba_opt_dict)
def numba_parallel(a, b, c):

    for i in nb.prange(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] += a[i,k]*b[k,j]

@nb.jit(parallel=True,**numba_opt_dict)
def numba_parallel_noreduce(a, b, c):

    for i in nb.prange(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] = c[i,j] + a[i,k]*b[k,j]

def numpy_matmul(a, b, c):
    np.matmul(a, b, out=c)

dtypes_dict = {
    "float32" : np.float32,
    "float64" : np.float64,
}

def setup_abc(P, R, Q, real_dtype):
    a = np.random.random((P,R)).astype(dtype=dtypes_dict[real_dtype])
    b = np.random.random((R,Q)).astype(dtype=dtypes_dict[real_dtype])
    c = np.zeros((P,Q),dtype=dtypes_dict[real_dtype])
    return {'a':a, 'b':b, 'c':c}

basename = 'matmul_timings_full'
filename = os.path.join(timings_folder,basename+'.npz')

all_args = {
    "P" : [(2 ** k) for k in range(15)]     ,
    "Q" : [(2 ** k) for k in range(15)]     ,
    "R" : [(2 ** k) for k in range(15)]     ,
    "real_dtype": ["float32", "float64"]    ,
}

all_funs = [
    python                  ,
    numba_serial            ,
    numba_auto_parallel     ,
    numba_parallel          ,
    numba_parallel_noreduce ,
    numpy_matmul                        ,
]

n_repeat = 1

MonotonicAxes = ["P", "Q", "R"]

time_per_test = 0.02

# %%

all_timings = pyquickbench.run_benchmark(
    all_args                        ,
    all_funs                        ,
    setup = setup_abc               ,
    filename = filename             ,
    StopOnExcept = True             ,
    ShowProgress = True             ,
    n_repeat = n_repeat             ,
    time_per_test = time_per_test   ,
    MonotonicAxes = MonotonicAxes   ,
    PreventBenchmark = True         ,
)

    
default_ax_val = 2**5

for ax_name in ["P","Q","R"]:

    plot_intent = {
        "P" : 'single_value'                        ,
        "Q" : 'single_value'                        ,
        "R" : 'single_value'                        ,
        "real_dtype" : "curve_linestyle"            ,
        pyquickbench.fun_ax_name :  'single_value'  ,
    }
    
    plot_intent[ax_name] = "points"

    single_values_val = {
        "P" : default_ax_val        ,
        "Q" : default_ax_val        ,
        "R" : default_ax_val        ,
        "real_dtype": "float64"     ,
        pyquickbench.fun_ax_name : "numpy_matmul"        ,
    }

    pyquickbench.plot_benchmark(
        all_timings                             ,
        all_args                                ,
        all_funs                                ,
        plot_intent = plot_intent               ,
        single_values_val = single_values_val   ,
        show = True                             ,
    )



    # %%

    pyquickbench.plot_benchmark(
        all_timings                             ,
        all_args                                ,
        all_funs                                ,
        plot_intent = plot_intent               ,
        single_values_val = single_values_val   ,
        logx_plot = True                        ,
        show = True                             ,
    )
