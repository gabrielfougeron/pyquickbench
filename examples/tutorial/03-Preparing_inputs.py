"""
Preparing inputs
================
"""

# %% 
# By default, the functions to be benchmarked with :func:`pyquickbench.run_benchmark` are expected to take as arguments a single integer ``n`` as in the following code.

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

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

import pyquickbench

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')
basename = f'Preparing_inputs_n'
timings_filename = os.path.join(timings_folder, basename+'.npy')

# sphinx_gallery_end_ignore

def comprehension(n):
    return ['' for _ in range(n)]

def star_operator(n):
    return ['']*n

def for_loop_append(n):
    l = []
    for _ in range(n):
        l.append('')
    
all_funs = [
    comprehension   ,
    star_operator   ,
    for_loop_append ,
]

n_bench = 12
all_sizes = [2**n for n in range(n_bench)]

pyquickbench.run_benchmark(
    all_sizes   ,
    all_funs    ,
    show = True ,
# sphinx_gallery_start_ignore
    filename = timings_filename     ,
# sphinx_gallery_end_ignore
) 


# %%
# Most often however, the functions to be benchmarked take data as input that is more complex than a simple integer and a setup phase is needed. In the following example, we want to compare different implementations of array summation algorithms.
#

import numpy as np
import math

def builtin_sum(x):
    return sum(x)

def np_sum(x):
    return np.sum(x)

def m_fsum(x):
    return math.fsum(x)

all_funs = [
    builtin_sum ,
    np_sum      ,
    m_fsum      ,
]

# sphinx_gallery_start_ignore
timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')
basename = f'Preparing_inputs_x'
timings_filename = os.path.join(timings_folder, basename+'.npy')
# sphinx_gallery_end_ignore

n_bench = 12
all_sizes = np.array([2**n for n in range(n_bench)])

# %%
# Here, we want the benchmark to measure the time taken by the different implementations as a function of the size of the input. The ``setup`` argument expects a callable that provides arguments for our different implementations to be timed. For instance, ``setup`` can return a dictionnnary ``kwargs``, so that each function to be benchmarked will be called as ``fun(**kwargs)``.

def setup(n):
    x = np.random.random(n)
    return {'x' : x}

pyquickbench.run_benchmark(
    all_sizes       ,
    all_funs        ,
    setup = setup   ,
    show = True     ,
# sphinx_gallery_start_ignore
    filename = timings_filename ,
# sphinx_gallery_end_ignore
) 
# %%
#  This calling convention is why positional-only arguments are disallowed in :func:`pyquickbench.run_benchmark`. For instance, even though the following defines a legal Python function

def pos_only_fun(n,/):
    return n

print(pos_only_fun(42))

# %%
#  It is not allowed in :func:`pyquickbench.run_benchmark` since the following raises an error:

try:
    pos_only_fun(n=42)
except TypeError as err:
    print(f'TypeError: {err}')
    