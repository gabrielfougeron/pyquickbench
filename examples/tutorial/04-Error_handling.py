"""
Handling errors
===============
"""

# %% 
# By default, :func:`pyquickbench.run_benchmark` will try to benchmark as much as possible even if the callables to be benchmarked throw errors. These errors are caught and the corresponding value in the benchmark is recorded as ``np.nan``, which will in turn show in plots as a missing value.

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

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)
    
timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')
basename = f'Error_handling'
timings_filename = os.path.join(timings_folder, basename+'.npy')

# sphinx_gallery_end_ignore

import pyquickbench

def comprehension(n):
    if n == 1 :
        raise ValueError('Forbidden value')
    return ['' for _ in range(n)]

def star_operator(n):
    if n == 8:
        raise ValueError('Forbidden value')
    return ['']*n

def for_loop_append(n):
    if n == 32:
        raise ValueError('Forbidden value')
    l = []
    for _ in range(n):
        l.append('')
    
all_funs = [
    comprehension   ,
    star_operator   ,
    for_loop_append ,
]

n_bench = 8
all_sizes = np.array([2**n for n in range(n_bench)])

pyquickbench.run_benchmark(
    all_sizes   ,
    all_funs    ,
    show = True ,
# sphinx_gallery_start_ignore
    filename = timings_filename ,
# sphinx_gallery_end_ignore
) 

# %% This default can be overriden with the argument ``StopOnExcept`` set to ``True``.

try:
    pyquickbench.run_benchmark(
        all_sizes           ,
        all_funs            ,
        show = True         ,
        StopOnExcept = True ,
    ) 
except Exception as exc:
    print(f'Exception thrown: {exc}') 