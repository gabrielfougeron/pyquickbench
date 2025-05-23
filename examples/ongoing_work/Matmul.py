"""
Summing elements of a numpy array
=================================
"""

# %% 
# This benchmark compares accuracy and efficiency of several summation algorithms in floating point arithmetics

# sphinx_gallery_start_ignore

import os
import sys
import threadpoolctl
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
threadpoolctl.threadpool_limits(limits=1).__enter__()

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

import matplotlib.pyplot as plt
import numpy as np
import math as m
import numba as nb
import torch
import warnings
warnings.filterwarnings("ignore")

print(f'{torch.cuda.is_available() = }')
print(f'{torch.cuda.get_device_name(0) = }')

device = torch.device("cuda")

numba_opt_dict = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

def python_impl(a, b, c):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] += a[i,k]*b[k,j]

numba_serial_impl = nb.jit(python_impl,**numba_opt_dict)

@nb.jit(parallel=True,**numba_opt_dict)
def numba_parallel_impl(a, b, c):

    for i in nb.prange(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):

                c[i,j] += a[i,k]*b[k,j]
                
def python(a, b, c, a_cu, b_cu, c_cu):
    return python_impl(a, b, c)
             
def numba_serial(a, b, c, a_cu, b_cu, c_cu):
    numba_serial_impl(a,b,c)

def numba_parallel(a, b, c, a_cu, b_cu, c_cu):
    numba_parallel_impl(a,b,c)    

def numpy_matmul(a, b, c, a_cu, b_cu, c_cu):
    np.matmul(a, b, out=c)
    
def torch_matmul(a, b, c, a_cu, b_cu, c_cu):
    torch.matmul(a_cu, b_cu, out=c_cu)

np_dtypes_dict = {
    "float16" : np.float16,
    "float32" : np.float32,
    "float64" : np.float64,
}

torch_dtypes_dict = {
    "float16" : torch.float16,
    "float32" : torch.float32,
    "float64" : torch.float64,
}

def setup_abc(P, R, Q, real_dtype):
    
    a = np.random.random((P,R)).astype(dtype=np_dtypes_dict[real_dtype])
    b = np.random.random((R,Q)).astype(dtype=np_dtypes_dict[real_dtype])
    c = np.zeros((P,Q),dtype=np_dtypes_dict[real_dtype])
    
    a_cu = torch.rand((P,R), dtype=torch_dtypes_dict[real_dtype], device=device)
    b_cu = torch.rand((R,Q), dtype=torch_dtypes_dict[real_dtype], device=device)
    c_cu = torch.zeros((P,Q), dtype=torch_dtypes_dict[real_dtype], device=device)
    
    return {'a':a, 'b':b, 'c':c, 'a_cu':a_cu, 'b_cu':b_cu, 'c_cu':c_cu}

def setup_abc_square(N, real_dtype):
    return setup_abc(N, N, N, real_dtype)

basename = 'matmul_timings_full'
filename = os.path.join(timings_folder,basename+'.npz')

all_args = {
    "N" : [(2 ** k) for k in range(20)]     ,
    "real_dtype": ["float16", "float32", "float64"]    ,
}

all_funs = [
    python                  ,
    numba_serial            ,
    numba_parallel          ,
    numpy_matmul            ,
    torch_matmul            ,
]

n_repeat = 1

MonotonicAxes = ["N"]

time_per_test = 0.02

# %%

all_timings = pyquickbench.run_benchmark(
    all_args                        ,
    all_funs                        ,
    setup = setup_abc_square        ,
    filename = filename             ,
    StopOnExcept = False             ,
    ShowProgress = True             ,
    n_repeat = n_repeat             ,
    time_per_test = time_per_test   ,
    MonotonicAxes = MonotonicAxes   ,
    # PreventBenchmark = True         ,
    # ForceBenchmark = True           ,
    WarmUp = True,
)


plot_intent = {
    "N" : 'points'                              ,
    "real_dtype" : "curve_linestyle"            ,
    pyquickbench.fun_ax_name :  'curve_color'   ,
}

single_values_val = {
    "real_dtype": "float64"     ,
}

pyquickbench.plot_benchmark(
    all_timings                             ,
    all_args                                ,
    all_funs                                ,
    plot_intent = plot_intent               ,
    single_values_val = single_values_val   ,
    show = True                             ,
)

