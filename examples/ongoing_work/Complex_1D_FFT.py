"""
Complex 1D FFT
==============
"""

# %% 
# This benchmark compares several implementations of complex 1-dimensionnal discrete Fourier Transform.
# 
import os
import threadpoolctl

# sphinx_gallery_start_ignore


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

threadpoolctl.threadpool_limits(limits=1).__enter__()

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
    
DP_Wisdom_file = os.path.join(timings_folder,"PYFFTW_wisdom.txt")
import pyfftw
import json

Load_wisdom = True
# Load_wisdom = False

if Load_wisdom:
    
    if os.path.isfile(DP_Wisdom_file):
        with open(DP_Wisdom_file,'r') as jsonFile:
            Wis_dict = json.load(jsonFile)
    
    wis = (
        Wis_dict["double"].encode('utf-8'),
        Wis_dict["single"].encode('utf-8'),
        Wis_dict["long"]  .encode('utf-8'),
    )

    pyfftw.import_wisdom(wis)

# sphinx_gallery_end_ignore

import numpy as np
import scipy
import mkl_fft
import ducc0
import pyfftw

import pyquickbench

def numpy_fft(fft_type, x, y, pyfftw_kwargs):
    if fft_type == "fft":
        np.fft.fft(x, axis=0)
    elif fft_type == "ifft":
        np.fft.ifft(x, axis=0)
    elif fft_type == "rfft":
        np.fft.rfft(x, axis=0)
    elif fft_type == "irfft":
        np.fft.irfft(x, axis=0)

def scipy_fft(fft_type, x, y, pyfftw_kwargs):
    if fft_type == "fft":
        scipy.fft.fft(x, axis=0,workers=1)
    elif fft_type == "ifft":
        scipy.fft.ifft(x, axis=0,workers=1)
    elif fft_type == "rfft":
        scipy.fft.rfft(x, axis=0,workers=1)
    elif fft_type == "irfft":
        scipy.fft.irfft(x, axis=0,workers=1)

def mkl__fft(fft_type, x, y, pyfftw_kwargs):

    if fft_type == "fft":
        mkl_fft._numpy_fft.fft(x, axis=0)
    elif fft_type == "ifft":
        mkl_fft._numpy_fft.ifft(x, axis=0)
    elif fft_type == "rfft":
        mkl_fft._numpy_fft.rfft(x, axis=0)
    elif fft_type == "irfft":
        mkl_fft._numpy_fft.irfft(x, axis=0)
    
def ducc_fft(fft_type, x, y, pyfftw_kwargs):
    if fft_type == "fft":
        ducc0.fft.c2c(x, axes=[0],nthreads=1, forward=True)
    elif fft_type == "ifft":
        ducc0.fft.c2c(x, axes=[0],nthreads=1, forward=False)
    elif fft_type == "rfft":
        ducc0.fft.r2c(x, axes=[0],nthreads=1)
    elif fft_type == "irfft":
        ducc0.fft.c2r(x, axes=[0],nthreads=1)

def pyfftw_ESTIMATE(fft_type, x, y, pyfftw_kwargs):
    planner_effort = ('FFTW_ESTIMATE', )
    fft_object = pyfftw.FFTW(x, y, flags=planner_effort, **pyfftw_kwargs)
    fft_object()

def pyfftw_EXHAUSTIVE(fft_type, x, y, pyfftw_kwargs):
    planner_effort = ('FFTW_EXHAUSTIVE', 'FFTW_WISDOM_ONLY')
    fft_object = pyfftw.FFTW(x, y, flags=planner_effort, **pyfftw_kwargs)
    fft_object()

def setup(fft_type, n_exp, base_fac, stride):
    
    n = base_fac * 2**n_exp
    
    simd_n = pyfftw.simd_alignment
    
    if fft_type in ['fft', 'ifft']:
        m = n
        xdtype = 'complex128'
        ydtype = 'complex128'
    elif fft_type in ['rfft','irfft']:
        m = n//2 + 1
        xdtype = 'float64'
        ydtype = 'complex128'
        
    if fft_type in ['fft', 'rfft']:
        direction = 'FFTW_FORWARD'
    elif fft_type in ['ifft','irfft']:
        direction = 'FFTW_BACKWARD'
        m, n = n, m
        xdtype, ydtype = ydtype, xdtype
        
    x = pyfftw.empty_aligned((n, stride), dtype=xdtype, n=simd_n)
    y = pyfftw.empty_aligned((m, stride), dtype=ydtype, n=simd_n)

    if xdtype == 'complex128':
        x[:] = np.random.random((n, stride)) + 1j * np.random.random((n, stride))
    elif xdtype == 'float64':
        x[:] = np.random.random((n, stride))
        
    if ydtype == 'complex128':
        y[:] = np.random.random((m, stride)) + 1j * np.random.random((m, stride))
    elif ydtype == 'float64':
        y[:] = np.random.random((m, stride))

    else:
        raise ValueError(f'No prepare function for {fft_type}')
    
    pyfftw_kwargs = {
        "axes" : (0, )              ,
        "direction" : direction     ,
        "threads":1                 ,
        'planning_timelimit':None   ,
    }
    
    return {'fft_type': fft_type, 'x': x, 'y' : y, 'pyfftw_kwargs':pyfftw_kwargs}


# sphinx_gallery_start_ignore

basename = 'complex_1D_fft'
timings_filename = os.path.join(timings_folder,basename+'.npz')

# sphinx_gallery_end_ignore

base_fac_list = [128]
# base_fac_list = [2*n+1 for n in range(13)]
# base_fac_list = list(range(100))

n_exp_min = 0
# n_exp_max = n_exp_min
n_exp_max = 10

all_strides = [1]
# all_strides = [1, 2, 4, 8]

all_fft_types = ["fft","ifft","rfft","irfft"]

all_args = {
    "fft_type": all_fft_types,
    "n_exp": np.array(range(n_exp_min, n_exp_max+1), dtype=np.intp) ,
    "base_fac" : np.array(base_fac_list, dtype=np.intp)             ,
    "stride" : np.array(all_strides, dtype=np.intp)             ,
}

all_funs = [
    numpy_fft           ,
    scipy_fft           ,
    mkl__fft            ,
    ducc_fft            ,
    pyfftw_ESTIMATE     ,
    pyfftw_EXHAUSTIVE   ,
]

# %%

all_timings = pyquickbench.run_benchmark(
    all_args                        ,
    all_funs                        ,
    setup = setup                   ,
    filename = timings_filename     ,
    ShowProgress = True             ,
    StopOnExcept = True             ,
    ForceBenchmark = True           ,
    # time_per_test = 0               ,
    MonotonicAxes = ["n_exp"]       ,
)

plot_intent = {
    "fft_type"                  : "subplot_grid_y"  ,
    "n_exp"                     : "points"          ,
    "base_fac"                  : "subplot_grid_y"  ,
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
    all_timings                 ,
    all_args                    ,
    all_funs                    ,
    plot_intent = plot_intent   ,
    show = True                 ,
    # alpha = 0.3                 ,
    relative_to_val = relative_to_val   ,
)
