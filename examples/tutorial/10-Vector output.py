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
    vec = rng.random((n))
    vec.sort()
    
    l = [vec[nsm*i] for i in range(m)]
    l.append(vec[-1])
    
    return np.array(l)
    
m = 10
uniform_decile = functools.partial(uniform_quantiles, m=m)
uniform_decile.__name__ = "uniform_decile"
    
all_funs = [
    uniform_decile   ,   
]

n_bench = 12
all_sizes = [10*2**n for n in range(n_bench)]

n_repeat = 100
    
all_values = pyquickbench.run_benchmark(
    all_sizes                   ,
    all_funs                    ,
    n_out = m+1                 , 
    n_repeat = n_repeat         ,
    mode = "vector_output"      ,
    StopOnExcept = True         ,

) 

plot_intent = {
    pyquickbench.default_ax_name : "points"         ,   
    pyquickbench.fun_ax_name : "curve_color"        ,   
    pyquickbench.repeat_ax_name : "same",   
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
    # title = "Computational cost growth order"   ,
    # transform = "pol_growth_order"              ,
)

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    plot_intent = plot_intent       ,
    show = True                     ,
    transform = "pol_growth_order"              ,
    # title = "Computational cost growth order"   ,
    # transform = "pol_growth_order"              ,
)
