"""
Numpy array initialization
==========================
"""

# %% 
# This benchmark compares the execution time of several :class:`numpy:numpy.ndarray` initialization routines.

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

numba_opt_dict = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)
    
basename = 'numpy_array_init'
filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

import numpy as np
import pyquickbench

dtypes_dict = {
    "float32" : np.float32,
    "float64" : np.float64,
}

def zeros(n, real_dtype):
    np.zeros((n), dtype=dtypes_dict[real_dtype])
    
def ones(n, real_dtype):
    np.ones((n), dtype=dtypes_dict[real_dtype])
    
def empty(n, real_dtype):
    np.empty((n), dtype=dtypes_dict[real_dtype])
    
def full(n, real_dtype):
    np.full((n), 0., dtype=dtypes_dict[real_dtype])

all_funs = [
    zeros   ,
    ones    ,
    empty   ,
    full    ,
]

all_args = {
    "n" : np.array([2**n for n in range(0, 30)]),
    "real_dtype" : ["float32", "float64"],
}

def setup(n, real_dtype):
    return {'n': n, 'real_dtype':real_dtype}

# %%

all_timings = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup           ,
    filename = filename     ,
    StopOnExcept = True     ,
    ShowProgress = True     ,

)

plot_intent = {
    "n" : 'points'                              ,
    "real_dtype" : 'curve_linestyle'            ,
    pyquickbench.fun_ax_name :  'curve_color'   ,
}

pyquickbench.plot_benchmark(
    all_timings                             ,
    all_args                                ,
    all_funs                                ,
    plot_intent = plot_intent               ,
    show = True                             ,
)

# %%
# While these measurement seem surprizing, they are explained in `the numpy documentation <https://numpy.org/doc/stable/benchmarking.html>`_ :
# 
# 
#       Be mindful that large arrays created with ``np.empty`` or ``np.zeros`` might not be allocated in physical memory until the memory is accessed. [...] One can force pagefaults to occur in the setup phase either by calling ``np.ones`` or ``arr.fill(value)`` after creating the array.
# 
# 
