"""
Time Trains
===========
"""

# %% 
# Quite 

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

def uniform_quantiles(n, m):
   
    TT = pyquickbench.TimeTrain(
        include_locs = False,
    )
    
    vec = np.random.random((n+1))
    TT.toc("Random Simulation")
    
    vec.sort()
    TT.toc("Sorting")

    return TT
    
m = 10
uniform_decile = functools.partial(uniform_quantiles, m=m)
uniform_decile.__name__ = "uniform_decile"
    
all_funs = [
    uniform_decile   ,   
]

n_bench = 20
all_sizes = [m * 2**n for n in range(n_bench)]

n_repeat = 100
    
plot_intent = {
    pyquickbench.default_ax_name : "points"         ,   
    pyquickbench.repeat_ax_name : "reduction_min"   ,   
    pyquickbench.out_ax_name : "curve_color"        ,   
}

    
pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    n_repeat = n_repeat             ,
    mode = "vector_output"          ,
    StopOnExcept = True             ,
    plot_intent = plot_intent       ,
    show = True                     ,
) 
