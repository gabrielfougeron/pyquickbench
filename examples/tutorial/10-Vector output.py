"""
TimeTrains
==========
"""

# %% 



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
    
import numpy as np
import math as m
import functools

# sphinx_gallery_end_ignore

import pyquickbench

def uniform_quantiles(n,m):
    
    assert n % m == 0
    
    nsm = n // m
    
    rng = np.random.default_rng()
    vec = rng.random((n+1))
    vec.sort()
    
    l = [vec[nsm*i]for i in range(m+1)]
    
    return np.array(l)
    
m = 4
uniform_decile = functools.partial(uniform_quantiles, m=m)
uniform_decile.__name__ = "uniform_decile"
    
all_funs = [
    uniform_decile   ,   
]

n_bench = 20
all_sizes = [2**n for n in range(5,n_bench)]

n_repeat = 1
    
all_values = pyquickbench.run_benchmark(
    all_sizes                   ,
    all_funs                    ,
    n_repeat = n_repeat         ,
    mode = "vector_output"      ,
    StopOnExcept = True         ,
    nproc = 16                  ,

) 

plot_intent = {
    pyquickbench.default_ax_name : "points"         ,   
    pyquickbench.fun_ax_name : "curve_color"        ,   
    pyquickbench.repeat_ax_name : "reduction_median",   
    pyquickbench.out_ax_name : "curve_color"        ,   
}

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    plot_intent = plot_intent       ,
    show = True                     ,
    logy_plot = False               ,
    plot_ylim = (0.,1.)             ,
    # alpha = 100./255                  ,
    # title = "Computational cost growth order"   ,
    # transform = "pol_growth_order"              ,
)


# %% 


def uniform_quantiles_error(n,m):
    
    vec = np.random.random((n+1))
    vec.sort()
    
    return np.array([abs(vec[(n // m)*i] - i / m) for i in range(m+1)])

uniform_decile_error = functools.partial(uniform_quantiles_error, m=m)
uniform_decile_error.__name__ = "uniform_decile_error"

all_funs = [
    uniform_decile_error   ,   
]

n_repeat = 1000

all_errors = pyquickbench.run_benchmark(
    all_sizes                   ,
    all_funs                    ,
    n_repeat = n_repeat         ,
    mode = "vector_output"      ,
    StopOnExcept = True         ,
    nproc = 16                  ,

) 

plot_intent = {
    pyquickbench.default_ax_name : "points"         ,   
    pyquickbench.fun_ax_name : "curve_color"        ,   
    pyquickbench.repeat_ax_name : "reduction_avg"   ,   
    pyquickbench.out_ax_name : "curve_color"        ,   
}

all_out_names = ["1st", "2nd", "3rd", "4th", "5th", ]

pyquickbench.plot_benchmark(
    all_errors                      ,
    all_sizes                       ,
    all_funs                        ,
    plot_intent = plot_intent       ,
    show = True                     ,
    ylabel = "Estimator error"      ,
    all_out_names = all_out_names,
    # title = "Computational cost growth order"   ,
    # transform = "pol_growth_order"              ,
)

pyquickbench.plot_benchmark(
    all_errors                      ,
    all_sizes                       ,
    all_funs                        ,
    plot_intent = plot_intent       ,
    show = True                     ,
    transform = "pol_growth_order"  ,
    # title = "Computational cost growth order"   ,
    # transform = "pol_growth_order"              ,
)
