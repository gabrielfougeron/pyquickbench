"""
Complex 1D FFT
==============
"""

# %% 
# This benchmark compares several implementations of complex 1-dimensionnal discrete Fourier Transform.
# 
import os
import threadpoolctl

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['TBB_NUM_THREADS'] = '1'
# 
# threadpoolctl.threadpool_limits(limits=1).__enter__()

# sphinx_gallery_start_ignore

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

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

import numpy as np
import scipy
import mkl_fft
import ducc0

import pyquickbench

def numpy_fft(x):
    return np.fft.fft(x, axis=0)

def scipy_fft(x):
    return scipy.fft.fft(x, axis=0)

def mkl__fft(x):
    return mkl_fft._numpy_fft.fft(x, axis=0)

def ducc_fft(x):
    return ducc0.fft.c2c(x, axes=[0])

def setup(n_exp, base_fac, stride):
    n = base_fac * 2**n_exp
    return {'x': np.random.random((n, stride)) + 1j * np.random.random((n, stride))}


# sphinx_gallery_start_ignore

basename = 'complex_1D_fft'
timings_filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

# base_fac_list = [1]
base_fac_list = [2*n+1 for n in range(13)]
# base_fac_list = list(range(100))


n_exp_min = 5
# n_exp_max = n_exp_min
n_exp_max = 20

all_strides = [1]
# all_strides = [1, 2, 4, 8, 16, 32, 64]

all_args = {
    "n_exp": np.array(range(n_exp_min, n_exp_max+1), dtype=np.intp) ,
    "base_fac" : np.array(base_fac_list, dtype=np.intp)             ,
    "stride" : np.array(all_strides, dtype=np.intp)             ,
}

all_funs = [
    numpy_fft  ,
    scipy_fft  ,
    mkl__fft   ,
    ducc_fft   ,
]

# %%

all_errors = pyquickbench.run_benchmark(
    all_args                        ,
    all_funs                        ,
    setup = setup                   ,
    filename = timings_filename     ,
    ShowProgress = True             ,
    StopOnExcept = True             ,
    # ForceBenchmark = True           ,
    WarmUp = True                   ,
    MonotonicAxes = ["n_exp"]       ,
)

plot_intent = {
    "n_exp"                     : "points"          ,
    "base_fac"                  : "subplot_grid_y",
    "stride"                    : "subplot_grid_y"  ,
    pyquickbench.fun_ax_name    : "curve_color"     ,
}

# plot_intent = {
#     "n_exp"                     : "subplot_grid_y"          ,
#     "base_fac"                  : "subplot_grid_y",
#     "stride"                    : "points"  ,
#     pyquickbench.fun_ax_name    : "curve_color"     ,
# }

# plot_intent = {
#     "n_exp"                     : "points"          ,
#     "base_fac"                  : "subplot_grid_y",
#     "stride"                    : "curve_pointstyle"  ,
#     pyquickbench.fun_ax_name    : "curve_color"     ,
# }

# plot_intent = {
#     "n_exp"                     : "subplot_grid_y"  ,
#     "base_fac"                  : "points"          ,
#     "stride"                    : "subplot_grid_y"  ,
#     pyquickbench.fun_ax_name    : "curve_color"     ,
# }



relative_to_val = None
# 
# relative_to_val = {
#     "stride" : 8    ,
# }


pyquickbench.plot_benchmark(
    all_errors                  ,
    all_args                    ,
    all_funs                    ,
    plot_intent = plot_intent   ,
    show = True                 ,
    # alpha = 0.3                 ,
    relative_to_val = relative_to_val   ,
)
