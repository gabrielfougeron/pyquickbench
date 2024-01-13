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


PRIMES = np.array([
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419
])

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return 0.

    for i in range(2, n):
        if n % i == 0:
            return 0.
    return 1.

is_prime_nb = nb.jit(is_prime,**numba_opt_dict)

def is_prime_nb_wrapped(n):
    m = n+3
    
    toto = is_prime_nb(n) + 1
    
    return toto
    

# sphinx_gallery_start_ignore

basename = 'primes_output'
filename = os.path.join(timings_folder,basename+'_new_________.npz')

# sphinx_gallery_end_ignore

all_funs = [
    # is_prime                  ,
    is_prime_nb            ,
    # is_prime_nb_wrapped            ,
]

# %%

all_errors = pyquickbench.run_benchmark(
    PRIMES                ,
    all_funs                ,
    filename = filename     ,
    StopOnExcept = True     ,
    ShowProgress = True     ,
    mode = "scalar_output"          ,
    # PreventBenchmark = True ,
    ForceBenchmark= True    ,
    nproc = 6   ,
    # pooltype = "phony"   ,
    pooltype = "thread"   ,
    # pooltype = "process"   ,
)

# plot_intent = {
#     "P" : 'single_value'                  ,
#     "Q" : 'points'            ,
#     "R" : 'single_value'            ,
#     "real_dtype": 'curve_linestyle'  ,
# }
# 
# single_values_idx = {
#     "P" : -1        ,
#     "Q" : -1        ,
#     "R" : -1        ,
#     "real_dtype": 0 ,
# }
# 
# pyquickbench.plot_benchmark(
#     all_errors                              ,
#     all_args                                ,
#     all_funs                                ,
#     plot_intent = plot_intent               ,
#     single_values_idx = single_values_idx   ,
#     show = True                             ,
# )