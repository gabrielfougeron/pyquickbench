"""
Multithreaded benchmarks
========================
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

n_repeat = 1000
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
    "n"         : "points"          ,
    "fun"       : "curve_color"     ,
    "repeat"    : "same"            ,
}

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    show = True                     ,
    plot_intent =   plot_intent     ,
)


# %% 
# The above plot is quite cluttered. For more concise information, let's use curve transparency:

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    show = True                     ,
    plot_intent =   plot_intent     ,
    alpha = 1./255                  ,
)

# %% 
# The above plot gives a good idea of the concentration of data, but bounds on timing are not very clear. Using reductions in ``plot_intent``, we can choose to plot minimal, median and maximal values:
# 

dpi = 150
figsize = (1600/dpi, 800/dpi)

fig, ax = plt.subplots(
    nrows   = 1         ,
    ncols   = 1         ,
    figsize = figsize   ,
    dpi     = dpi       ,
    squeeze = False     ,
)

all_repeat_intents = ["reduction_min", "reduction_max", "reduction_median"]
all_linestyles = ["dotted", "dashed", "solid"]

for (
    repeat_intent       ,
    linestyle           ,
)in zip(
    all_repeat_intents  ,
    all_linestyles      ,
):

    plot_intent = {
        "n"         : "points"          ,
        "fun"       : "curve_color"     ,
        "repeat"    : repeat_intent     ,
    }

    pyquickbench.plot_benchmark(
        all_values                  ,
        all_sizes                   ,
        all_funs                    ,
        plot_intent = plot_intent   ,
        linestyle_list = linestyle  ,
        fig = fig                   ,
        ax = ax                     ,
    )
    
plt.tight_layout()
plt.show()


# %% 
# More generally, the ``plot_intent`` argument controls what dimension of the array ``all_values`` is plotted, and in what way. For instance, as a way to better understand the statistics of the measured timings, we can plot the measured time of execution as a function of the index of the repeated benchmark for a single function.

plot_intent = {
    "n"         : "curve_color"     , 
    "fun"       : "single_value"    ,
    "repeat"    : "points"          ,
}

single_values_val = {"fun": "star_operator"}

pyquickbench.plot_benchmark(
    all_values                              ,
    all_sizes                               ,
    all_funs                                ,
    show = True                             ,
    plot_intent =   plot_intent             ,
    single_values_val = single_values_val   ,
)

# %% 
# Or for all functions, but a single value of input size.

plot_intent = {
    "n"         : "reduction_max"   , 
    "fun"       : "curve_color"     ,
    "repeat"    : "points"          ,
}

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    show = True                     ,
    plot_intent =   plot_intent     ,
)


# %% 
# As can be seen in the above plots, the timings are automatically sorted along the "repeat" axis.