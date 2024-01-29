"""
Summing elements of a numpy array
=================================
"""

# %% 
# This benchmark compares accuracy and efficiency of several summation algorithms in floating point arithmetics

# sphinx_gallery_start_ignore

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

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
import math
import scipy


import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore


def np_sum(x):
    return np.sum(x)

def rfft(x):
    return scipy.fft.rfft(x)

def setup(n):
    return {'x': np.random.random((n))}

# sphinx_gallery_start_ignore

basename = 'sum_vs_fft'
timings_filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

all_args = {
    "n": np.array([2**i for i in range(27)]),
}

all_funs = [
    np_sum  ,
    rfft    ,
]

# %%



all_errors = pyquickbench.run_benchmark(
    all_args                    ,
    all_funs                    ,
    setup = setup               ,
    filename = timings_filename   ,
    ShowProgress = True,
)

relative_to_val = {
    pyquickbench.fun_ax_name : 'np_sum'  ,
}


pyquickbench.plot_benchmark(
    all_errors      ,
    all_args        ,
    all_funs        ,
    show = True
)

# %%
#

fig, ax = pyquickbench.plot_benchmark(
    all_errors      ,
    all_args        ,
    all_funs        ,
    relative_to_val = relative_to_val,
    # show = True
)

ax[0,0].plot(all_args["n"],np.log(all_args["n"])/np.log(2))

fig.tight_layout()
fig.show()