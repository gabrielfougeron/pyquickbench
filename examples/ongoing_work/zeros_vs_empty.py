"""
Summing elements of a numpy array
=================================
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
import math as m
import scipy
import numba as nb

numba_opt_dict = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

def zeros(n):
    np.zeros((n), dtype=np.float64)
def empty(n):
    np.empty((n), dtype=np.float64)

# sphinx_gallery_start_ignore

basename = 'zeros_vs_empty'
filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

all_funs = [
    zeros   ,
    empty   ,
]


all_args = {
    "n" : np.array([2**n for n in range(0, 30)]),
}


# %%

all_timings = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    filename = filename     ,
    StopOnExcept = True     ,
    ShowProgress = True     ,
    ForceBenchmark= True    ,

)

plot_intent = {
    "n" : 'points'                           ,
    pyquickbench.fun_ax_name :  'curve_color'       ,
}

relative_to_val_list = [
    None    ,
    {pyquickbench.fun_ax_name : 'empty'},
]

for relative_to_val in relative_to_val_list:

    pyquickbench.plot_benchmark(
        all_timings                             ,
        all_args                                ,
        all_funs                                ,
        plot_intent = plot_intent               ,
        show = True                             ,
        relative_to_val = relative_to_val       ,
    )

