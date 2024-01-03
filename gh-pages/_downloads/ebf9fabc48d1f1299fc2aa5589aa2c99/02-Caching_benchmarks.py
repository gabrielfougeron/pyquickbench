"""
Caching benchmarks
==================
"""

# %% 
# Performance benchmarks as run by pyquickbench are typically quite lengthy. Indeed, by default, pyquickbench uses the standard :meth:`python:timeit.Timer.autorange` to assess the number of times a benchmark should be run to ensure reliable timings, which comes with significant overhead as any call to this function will require a non-configurable minimum execution time of 0.2 seconds.
#
# In pyquickbench, caching results in order to avoid a full re-run is as simple as providing a file name to :func:`pyquickbench.run_benchmark`.
#  

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
    
timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')
basename = f'Caching_benchmarks'
timings_filename = os.path.join(timings_folder, basename+'.npy')

# sphinx_gallery_end_ignore

import pyquickbench

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
    filename = timings_filename     ,
) 

