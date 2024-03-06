"""
Multidimensional benchmarks
===========================
"""

# %% 
# One of pyquickbench's strengths is its ability to run multidimensional benchmarks to test function behavior changes with respect to several different arguments, or to assess repeatability of a benchmark.
#
# For instance, let's run the following benchmark a thousand times.
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
timings_filename = os.path.join(timings_folder, basename+'.npz')

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

n_repeat = 3
time_per_test = 0.2

all_values = pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    n_repeat = n_repeat             ,
    time_per_test = time_per_test   ,
    filename = timings_filename     ,
) 

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    show = True                     ,
)

# %% 
# By default, only the minminum timing is reported on the plot as recommended by :meth:`python:timeit.Timer.repeat`. This being said, the array ``all_values`` does contain ``n_repeat`` timings.
#

print(all_values.shape[0] == len(all_sizes))
print(all_values.shape[1] == len(all_funs))
print(all_values.shape[2] == n_repeat)

# %% 
# All the different timings can be superimposed on the same plot with the following ``plot_intent`` argument:

plot_intent = {
    pyquickbench.default_ax_name   : "points"       ,
    pyquickbench.fun_ax_name       : "curve_color"  ,
    pyquickbench.repeat_ax_name    : "same"         ,
}

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    show = True                     ,
    plot_intent =   plot_intent     ,
)

relative_to_val = {
    pyquickbench.fun_ax_name: "star_operator" ,
}

pyquickbench.plot_benchmark(
    all_values                          ,
    all_sizes                           ,
    all_funs                            ,
    show = True                         ,
    plot_intent =   plot_intent         ,
    relative_to_val = relative_to_val   ,
)
