"""
A first benchmark
=================
"""

# %% 
# Let's run and plot a first simple benchmark in Python.
#
# Suppose I want to compare the efficiency of a few different methods to pre-allocate memory for a list of strings in Python.
#
# Let's define a separate python function for three different list pre-allocation strategies.
# These functions all take an integer called ``n`` as an input, which stands for the length of the list to be pre-allocated.
# The argument name ``n`` is the :data:`pyquickbench.default_ax_name` and can be changed in as described in :ref:`sphx_glr__build_auto_examples_tutorial_03-Preparing_inputs.py`.
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
basename = 'First_benchmark'
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

pyquickbench.run_benchmark(
    all_sizes   ,
    all_funs    ,
    show = True ,
# sphinx_gallery_start_ignore
    filename = timings_filename     ,
# sphinx_gallery_end_ignore
) 

# %% 
# From this benchmark, it is easy to see that initializing a list to a given length filled with a single value is quickest using the star operator.