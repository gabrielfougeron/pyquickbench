"""
Convergence analysis of scipy's Runge-Kutta methods for ODE IVP
===============================================================
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

def ODE_define_test(eq_name):
     
    if eq_name == "y'' = -y" :
        # WOLFRAM
        # y'' = - y
        # y(x) = A cos(x) + B sin(x)

        test_ndim = 2

        ex_sol = lambda t : np.array( [ np.cos(t) , np.sin(t),-np.sin(t), np.cos(t) ]  )

        fun = lambda t,y:   np.asarray(y)
        gun = lambda t,x:  -np.asarray(x)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] =  xy[2]
            fxy[1] =  xy[3]
            fxy[2] = -xy[0]
            fxy[3] = -xy[1]
            
            return fxy

    if eq_name == "y'' = - exp(y)" :
        # WOLFRAM
        # y'' = - exp(y)
        # y(x) = - 2 * ln( cosh(t / sqrt(2) ))

        test_ndim = 1

        invsqrt2 = 1./np.sqrt(2.)
        sqrt2 = np.sqrt(2.)
        ex_sol = lambda t : np.array( [ -2*np.log(np.cosh(invsqrt2*t)) , -sqrt2*np.tanh(invsqrt2*t) ]  )

        fun = lambda t,y:  np.array(y)
        gun = lambda t,x: -np.exp(x)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] = xy[1]
            fxy[1] = -np.exp(xy[0])

            return fxy

    if eq_name == "y'' = xy" :

        # Solutions: Airy functions
        # Nonautonomous linear test case

        test_ndim = 2

        def ex_sol(t):

            ai, aip, bi, bip = scipy.special.airy(t)

            return np.array([ai,bi,aip,bip])

        fun = lambda t,y: np.array(y)
        gun = lambda t,x: np.array([t*x[0],t*x[1]],dtype=np.float64)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] =  xy[2]
            fxy[1] =  xy[3]
            fxy[2] = t*xy[0]
            fxy[3] = t*xy[1]
            
            return fxy
        
        
    if eq_name == "y' = Az; z' = By" :

        test_ndim = 10

        A = np.diag(np.array(range(test_ndim)))
        B = np.identity(test_ndim)

        AB = np.zeros((2*test_ndim,2*test_ndim))
        AB[0:test_ndim,test_ndim:2*test_ndim] = A
        AB[test_ndim:2*test_ndim,0:test_ndim] = B

        yo = np.array(range(test_ndim))
        zo = np.array(range(test_ndim))

        yzo = np.zeros(2*test_ndim)
        yzo[0:test_ndim] = yo
        yzo[test_ndim:2*test_ndim] = zo

        def ex_sol(t):
            return scipy.linalg.expm(t*AB).dot(yzo)

        fun = lambda t,z: A.dot(z)
        gun = lambda t,y: B.dot(y)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[:test_ndim] =  A.dot(xy[test_ndim:])
            fxy[test_ndim:] =  B.dot(xy[:test_ndim])

            return fxy

    return fun, gun, fgun, ex_sol, test_ndim

def scipy_ODE_cpte_error_on_test(
    eq_name     ,
    method      ,
    nint        ,
):

    fun, gun, fgun, ex_sol, test_ndim = ODE_define_test(eq_name)

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

eq_names = [
    "y'' = -y"          ,
    "y'' = - exp(y)"    ,
    "y'' = xy"          ,
    "y' = Az; z' = By"  ,
]

method_names = [
    "RK45"  ,  
    "RK23"  ,  
    "DOP853",  
    "Radau" ,  
    "BDF"   ,  
    "LSODA" ,  
]

all_nint = np.array([2**i for i in range(12)])

all_benchs = {}
for eq_name in eq_names:
    bench = {}
    for method in method_names:
        
        bench[f'{method}'] = functools.partial(
            scipy_ODE_cpte_error_on_test ,
            eq_name ,
            method  ,     
        )
    
    all_benchs[eq_name] = bench

def setup(nint):
    return {'nint': nint}

n_bench = len(all_benchs)

dpi = 150
figsize = (1600/dpi, n_bench * 800 / dpi)

# %%
# The following plots give the measured relative error as a function of the number of quadrature subintervals

fig, axs = plt.subplots(
    nrows = n_bench,
    ncols = 1,
    sharex = True,
    sharey = False,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = False,
)

plot_ylim = [1e-17,1e1]

for i_bench, (bench_name, all_funs) in enumerate(all_benchs.items()):

    bench_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_error.npz')
    
    all_errors = pyquickbench.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
        StopOnExcept = True             ,
    )

    pyquickbench.plot_benchmark(
        all_errors                                  ,
        all_nint                                    ,
        all_funs                                    ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        plot_ylim = plot_ylim                       ,
        title = f'Relative error on integrand {bench_name}' ,
    )

plt.tight_layout()
plt.show()
    
plot_xlim = axs[0,0].get_xlim()



# %%
# Error as a function of running time

fig, axs = plt.subplots(
    nrows = n_bench,
    ncols = 1,
    sharex = False,
    sharey = False,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = False,
)

plot_ylim = [1e-17,1e1]

for i_bench, (bench_name, all_funs) in enumerate(all_benchs.items()):
    
    bench_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_error.npz') 

    all_errors = pyquickbench.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
    )
    
    timings_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_timings.npz') 
    
    all_times = pyquickbench.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "timings"                ,
        filename = timings_filename     ,
    )
    
    pyquickbench.plot_benchmark(
        all_errors                  ,
        all_nint                    ,
        all_funs                    ,
        all_xvalues = all_times     ,
        logx_plot = True            ,
        fig = fig                   ,
        ax = axs[i_bench,0]         ,
        plot_ylim = plot_ylim       ,
        title = f'Relative error as a function of computational cost for equation {bench_name}' ,
    )


plt.tight_layout()
plt.show()


