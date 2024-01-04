"""
Preparing inputs
================
"""

# %% 
# Most often, the functions to be benchmarked take data as input that is more complex than a simple integer. This data can
#
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
basename = f'Preparing_inputs'
timings_filename = os.path.join(timings_folder, basename+'.npy')

# sphinx_gallery_end_ignore

import numpy as np
import math

def builtin_sum(x):
    return sum(x)

def np_sum(x):
    return np.sum(x)

def m_fsum(x):
    return math.fsum(x)

all_funs = [
    builtin_sum ,
    np_sum      ,
    m_fsum      ,
]

def prepare_x(n):
    x = np.random.random(n)
    return [('x', x)]
 
# %% 
# Let's define relevant sizes of lists to be timed.

n_bench = 8
all_sizes = np.array([2**n for n in range(n_bench)])

# %% 
#
# Now, let's import pyquickbench, run and plot the benchmark

import pyquickbench

pyquickbench.run_benchmark(
    all_sizes   ,
    all_funs    ,
    show = True ,
    filename = timings_filename ,
) 
