"""
Sorting elements in an array
============================
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

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

class CountCmp():

    def __init__(self, cmp_lt=pyquickbench.sort.default_cmp_lt):
        self.n_cmp = 0
        self.cmp_lt = cmp_lt
    
    def __call__(self, a, b):
        self.n_cmp += 1
        return self.cmp_lt(a, b)
    
sort_name_to_fun = {
    "binary_insertion_sort" : pyquickbench.sort.binary_insertion_sort   ,
    "merge_sort"            : pyquickbench.sort.merge_sort              ,
    "heap_sort"             : pyquickbench.sort.heap_sort               ,
    "merge_insertion_sort"  : pyquickbench.sort.merge_insertion_sort    ,
    "quick_sort"            : pyquickbench.sort.quick_sort              ,
}
    
def count_inplace_sort_comparisons(arr, sortname):
    sort = sort_name_to_fun[sortname]
    arr_cp = arr.copy()
    cmp_counter = CountCmp()
    sort(arr_cp, cmp_counter)
    return cmp_counter.n_cmp

def setup(n, sortname):
    return {
        'arr': np.random.random(n)  ,
        'sortname': sortname        ,
    }

# sphinx_gallery_start_ignore

basename = 'sort_bench'
bench_filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

all_args = {
    "n" : np.array([n for n in range(2,200)])   ,
    "sort" : list(sort_name_to_fun.keys())      ,
}

# %%

n_repeat = 1000

all_vals = pyquickbench.run_benchmark(
    all_args                            ,
    [count_inplace_sort_comparisons]    ,
    setup = setup                       ,
    mode = "scalar_output"              ,
    n_repeat = n_repeat                 ,
    filename = bench_filename           ,
    StopOnExcept = True                 ,
    ShowProgress = True                 ,
    deterministic_setup = False         ,
    pooltype = "process"                ,
)

plot_intent = {
    "n"                         : 'points'          ,
    "sort"                      : 'curve_color'     ,
    pyquickbench.repeat_ax_name : 'reduction_avg'   ,
}

pyquickbench.plot_benchmark(
    all_vals                            ,
    all_args                            ,
    [count_inplace_sort_comparisons]    ,
    plot_intent = plot_intent           ,
    title = "Average number of comparisons needed to sort an array"   ,
    ylabel = "Average number of comparisons"    ,
    xlabel = "Size of the array"        ,
    show = True                         ,
)

relative_to_val = {"sort":"binary_insertion_sort"}

pyquickbench.plot_benchmark(
    all_vals                            ,
    all_args                            ,
    [count_inplace_sort_comparisons]    ,
    plot_intent = plot_intent           ,
    title = "Average number of comparisons needed to sort an array"   ,
    ylabel = "Average number of comparisons"    ,
    xlabel = "Size of the array"        ,
    show = True                         ,
    relative_to_val = relative_to_val   ,
)

# %%

plot_intent = {
    "n"                         : 'points'          ,
    "sort"                      : 'curve_color'     ,
    pyquickbench.repeat_ax_name : 'reduction_max'   ,
}

pyquickbench.plot_benchmark(
    all_vals                            ,
    all_args                            ,
    [count_inplace_sort_comparisons]    ,
    plot_intent = plot_intent           ,
    title = "Maximum number of comparisons needed to sort an array"   ,
    ylabel = "Max number of comparisons",
    xlabel = "Size of the array"        ,
    show = True                         ,
)

pyquickbench.plot_benchmark(
    all_vals                            ,
    all_args                            ,
    [count_inplace_sort_comparisons]    ,
    plot_intent = plot_intent           ,
    title = "Maximum number of comparisons needed to sort an array"   ,
    ylabel = "Max number of comparisons",
    xlabel = "Size of the array"        ,
    relative_to_val = relative_to_val   ,
    show = True                         ,
)
