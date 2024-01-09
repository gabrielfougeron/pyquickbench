"""
Plotting scalar values
======================
"""

# %%
# Evaluation of relative quadrature error with the following parameters:

# sphinx_gallery_start_ignore

import os
import sys
import itertools
import functools

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

import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'scipy_ivp_cvg_bench_'

# sphinx_gallery_end_ignore

def scipy_ODE_cpte_error_on_test(
    method      ,
    nint        ,
):

    # Solutions: Airy functions
    # Nonautonomous linear test case

    test_ndim = 2

    def ex_sol(t):

        ai, aip, bi, bip = scipy.special.airy(t)

        return np.array([ai,bi,aip,bip])

    def fgun(t, xy):
        
        fxy = np.empty(2*test_ndim)
        fxy[0] = xy[2]
        fxy[1] = xy[3]
        fxy[2] = t*xy[0]
        fxy[3] = t*xy[1]
        
        return fxy
        
    t_span = (0.,np.pi)
    
    max_step = (t_span[1] - t_span[0]) / nint

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    bunch = scipy.integrate.solve_ivp(
        fun = fgun                      ,
        t_span = t_span                 ,
        y0 = ex_init                    ,
        method = method                 ,
        t_eval = np.array([t_span[1]])  ,
        first_step = max_step           ,
        max_step = max_step             ,
        atol = 1.                       ,
        rtol = 1.                       ,
    )

    error = np.linalg.norm(bunch.y[:,0]-ex_final)/np.linalg.norm(ex_final)

    return error

method_names = [
    "RK45"  ,  
    "RK23"  ,  
    "DOP853",  
    "Radau" ,  
    "BDF"   ,  
    "LSODA" ,  
]

all_nint = np.array([2**i for i in range(12)])

bench = {}
for method in method_names:
    
    bench[f'{method}'] = functools.partial(
        scipy_ODE_cpte_error_on_test ,
        method  ,     
    )


def setup(nint):
    return {'nint': nint}

# %%
# The following plots give the measured relative error as a function of the number of quadrature subintervals

plot_ylim = [1e-17, 1e1]

bench_filename = os.path.join(bench_folder,basename_bench_filename+'_error.npz')

pyquickbench.run_benchmark(
    all_nint                        ,
    bench                           ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    filename = bench_filename       ,
    plot_ylim = plot_ylim                       ,
    title = f'Relative error on integrand'      ,
    ylabel = "Relative error"   ,
    show = True                                 ,
)


# %%
# Running time

timings_filename = os.path.join(bench_folder,basename_bench_filename+'_timings.npz') 

pyquickbench.run_benchmark(
    all_nint                        ,
    bench                           ,
    setup = setup                   ,
    mode = "timings"                ,
    filename = timings_filename     ,
    logx_plot = True                ,
    title = f'Computational cost'   ,
    show = True                     ,
)


# %%
# Error as a function of running time

bench_filename = os.path.join(bench_folder,basename_bench_filename+'_error.npz') 

all_errors = pyquickbench.run_benchmark(
    all_nint                        ,
    bench                           ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    filename = bench_filename       ,
)

timings_filename = os.path.join(bench_folder,basename_bench_filename+'_timings.npz') 

all_times = pyquickbench.run_benchmark(
    all_nint                        ,
    bench                           ,
    setup = setup                   ,
    mode = "timings"                ,
    filename = timings_filename     ,
)

pyquickbench.plot_benchmark(
    all_errors                  ,
    all_nint                    ,
    bench                       ,
    mode = "scalar_output"      ,
    all_xvalues = all_times     ,
    logx_plot = True            ,
    plot_ylim = plot_ylim       ,
    title = f'Relative error as a function of computational cost' ,
    ylabel = "Relative error"   ,
    xlabel = "Time (s)"         ,
    show = True                 ,
)
