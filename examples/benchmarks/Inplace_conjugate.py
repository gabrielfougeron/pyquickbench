"""
Benchmark of inplace conjugation of arrays
==========================================
"""

# %% 
#
# This is a benchmark of different ways to perform inplace conjugation of a complex numpy array.

# sphinx_gallery_start_ignore

import os
import sys
import multiprocessing
import itertools

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import numba as nb

numba_opt_dict = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

import pyquickbench

def numpy_ufunc_outofplace(x):
    x = np.conjugate(x)
    
def numpy_ufunc_inplace(x):
    np.conjugate(x, out=x)
    
def numpy_inplace_mul(x):
    x.imag *= -1

def numpy_subs(x):
    x.imag = -x.imag 
    
@nb.jit("void(complex128[::1])", **numba_opt_dict)
def numba_loop_typedef(x):
    
    for i in range(x.shape[0]):
        x.imag[i] = -x.imag[i]
        
@nb.jit(**numba_opt_dict)
def numba_loop(x):
    
    for i in range(x.shape[0]):
        x.imag[i] = -x.imag[i]
    
@nb.jit(**numba_opt_dict, parallel=True)
def numba_loop_parallel(x):
    
    for i in nb.prange(x.shape[0]):
        x.imag[i] = -x.imag[i]
  
all_funs = [
    numpy_ufunc_outofplace ,
    numpy_ufunc_inplace ,
    numpy_inplace_mul ,
    numpy_subs ,
    numba_loop_typedef ,
    numba_loop ,
    numba_loop_parallel ,
]

all_sizes = np.array([2**n for n in range(25)])

def prepare_x(n):
    x = np.random.random(n) + 1j*np.random.random(n)
    return [('x', x)]
    
basename = f'Inplace_conjugation_bench'
timings_filename = os.path.join(timings_folder, basename+'.npz')

n_repeat = 10

all_values = pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    setup = prepare_x               ,
    n_repeat = n_repeat             ,
    filename = timings_filename     ,
)

pyquickbench.plot_benchmark(
    all_values                      ,
    all_sizes                       ,
    all_funs                        ,
    show = True                     ,
    title = 'Inplace conjugation'   ,
)

# %% 
# 

relative_to_val = {pyquickbench.fun_ax_name:"numpy_ufunc_inplace"}

pyquickbench.plot_benchmark(
    all_values                          ,
    all_sizes                           ,
    all_funs                            ,
    relative_to_val = relative_to_val   ,
    show = True                         ,
    title = 'Inplace conjugation'       ,
    ylabel = 'Time relative to numpy_ufunc_inplace' ,
)


