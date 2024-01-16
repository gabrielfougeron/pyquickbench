"""
Multidimensional_benchmarks
===========================
"""

# %% 
# So far in the tutorial, 
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
basename = f'Muldidim_bench'
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
 
# %% 
# Let's define relevant sizes of lists to be timed.

n_bench = 12
all_sizes = [2**n for n in range(n_bench)]

# %% 
#
# Now, let's import pyquickbench, run and plot the benchmark

import pyquickbench

n_repeat = 1000
time_per_test = 0.2

pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    n_repeat = n_repeat             ,
    time_per_test = time_per_test   ,
    show = True                     ,
    filename = timings_filename     ,
    pooltype = 'process'            ,
) 



# %% 
# By default : minimum as in :meth:`python:timeit.Timer.repeat`
#

all_values = pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    n_repeat = n_repeat             ,
    time_per_test = time_per_test   ,
    filename = timings_filename     ,
) 

plot_intent = {
    "n"         : "points"          ,
    "fun"       : "curve_color"     ,
    "repeat"    : "same"   ,
}

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    show = True                     ,
    plot_intent =   plot_intent     ,
)

plot_intent = {
    "n"         : "points"          ,
    "fun"       : "curve_color"     ,
    "repeat"    : "reduction_min"   ,
}

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    show = True                     ,
    plot_intent =   plot_intent     ,
)
