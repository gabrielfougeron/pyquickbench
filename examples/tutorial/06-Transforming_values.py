"""
Plotting transformed values
===========================
"""

# %%
# Let's expand on the benchmark exposed in :ref:`sphx_glr__build_auto_examples_tutorial_05-Plotting_scalars.py`.
# The benchmark consists in understanding the convergence behavior of the following ODE integrators provided by :mod:`scipy:scipy`.

method_names = [
    "RK45"  ,  
    "RK23"  ,  
    "DOP853",  
    "Radau" ,  
]

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

def scipy_ODE_cpte_error_on_test(
    method      ,
    n           ,
):

    # y'' = - w**2 * y
    # y(x) = A cos(w*x) + B sin(w*x)

    test_ndim = 2
    
    w = 10

    ex_sol = lambda t : np.array( [ np.cos(w*t) , np.sin(w*t),-np.sin(w*t), np.cos(w*t) ]  )

    def fgun(t, xy):
        
        fxy = np.empty(2*test_ndim)
        fxy[0] =  w*xy[2]
        fxy[1] =  w*xy[3]
        fxy[2] = -w*xy[0]
        fxy[3] = -w*xy[1]
        
        return fxy
    
    t_span = (0.,np.pi)
    
    max_step = (t_span[1] - t_span[0]) / n

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

all_nint = np.array([2**i for i in range(16)])

bench = {
    method: functools.partial(
        scipy_ODE_cpte_error_on_test ,
        method  ,     
    ) for method in method_names
}

# sphinx_gallery_end_ignore

# %%
# Let's focus on timings first.
timings_filename = os.path.join(bench_folder,basename_bench_filename+'_timings.npz') 
timings_results = pyquickbench.run_benchmark(
    all_nint                                    ,
    bench                                       ,
    filename = timings_filename                 ,
)

pyquickbench.plot_benchmark(
    timings_results                 ,
    all_nint                        ,
    bench                           ,
    show = True                     ,
    title = 'Computational cost'    , 
)

# %%
# We can check that the integrations algorithms provided by :mod:`scipy:scipy` scale linearly with respect to the number of integrations steps with the ``transform`` argument set to ``"pol_growth_order"``:

pyquickbench.plot_benchmark(
    timings_results                             ,
    all_nint                                    ,
    bench                                       ,
    logx_plot = True                            ,
    title = "Computational cost growth order"   ,
    transform = "pol_growth_order"              ,
    show = True                                 ,
)

# %%
# Since all the functions in the benchmark have the same asymptotic behavior, it makes sense to compare then against each other. Let's pick a reference method, for instance "DOP853" and plot all timings relative to this reference. This is achieved with the ``relative_to_val`` argument.

relative_to_val = {pyquickbench.fun_ax_name: "DOP853"}

pyquickbench.plot_benchmark(
    timings_results                         ,
    all_nint                                ,
    bench                                   ,
    logx_plot = True                        ,
    title = "Relative computational cost"   ,
    relative_to_val = relative_to_val       ,
    show = True                             ,
)

# %%
# Let's focus on accuracy measurements now.

plot_ylim = [1e-17, 1e1]
scalar_filename = os.path.join(bench_folder,basename_bench_filename+'_error.npz')
bench_results = pyquickbench.run_benchmark(
    all_nint                                ,
    bench                                   ,
    mode = "scalar_output"                  ,
    filename = scalar_filename               ,
)

pyquickbench.plot_benchmark(
    bench_results                           ,
    all_nint                                ,
    bench                                   ,
    mode = "scalar_output"                  ,
    plot_ylim = plot_ylim                   ,
    title = f'Relative error on integrand'  ,
    ylabel = "Relative error"               ,
    show = True                             ,
)

# %%
# A natural question when assessing accuracy is to ask whether the measured convergence rates of the numerical methods match their theoretical convergence rates. This post-processing can be performed automatically by :func:`pyquickbench.run_benchmark` if the argument ``transform`` is set to ``"pol_cvgence_order"``.

plot_ylim = [0, 10]

pyquickbench.plot_benchmark(
    bench_results                           ,
    all_nint                                ,
    bench                                   ,
    mode = "scalar_output"                  ,
    plot_ylim = plot_ylim                   ,
    ylabel = "Measured convergence rate"    ,
    logx_plot = True                        ,
    transform = "pol_cvgence_order"         ,
    show = True                             ,
)


# %%
# The pre and post convergence behavior of the numerical algorithms really clutters the plots. In this case, a clearer plot is obtained if the argument ``clip_vals`` is set to ``True``.

pyquickbench.plot_benchmark(
    bench_results                           ,
    all_nint                                ,
    bench                                   ,
    mode = "scalar_output"                  ,
    plot_ylim = plot_ylim                   ,
    ylabel = "Measured convergence rate"    ,
    logx_plot = True                        ,
    transform = "pol_cvgence_order"         ,
    clip_vals = True                        ,
    show = True                             ,
)

# %%
# The measured values can now be compared to the theoretical convergence rates of the different methods. In order to plot your own data to the figure, you can handle the :class:`matplotlib:matplotlib.figure.Figure` and :class:`matplotlib:matplotlib.axes.Axes` objects yourself, And decoupe the calls to :func:`pyquickbench.run_benchmark` and :func:`pyquickbench.plot_benchmark` as shown here.

th_cvg_rate = {
    "RK45"  : 5,  
    "RK23"  : 3,  
    "DOP853": 8,  
    "Radau" : 5, 
}

fig, ax = plt.subplots(
    nrows = 1                       ,
    ncols = 1                       ,
    figsize = (1600/150, 800/150)   ,
    dpi = 150                       ,
    squeeze = False                 ,
)

pyquickbench.plot_benchmark(
    bench_results                           ,
    all_nint                                ,
    bench                                   ,
    mode = "scalar_output"                  ,
    fig = fig                               ,
    ax = ax                                 ,
    plot_ylim = plot_ylim                   ,
    ylabel = "Measured convergence rate"    ,
    logx_plot = True                        ,
    transform = "pol_cvgence_order"         ,
    clip_vals = True                        ,
)

xlim = ax[0,0].get_xlim()
for name in bench:
        
    th_order = th_cvg_rate[name]
    ax[0,0].plot(xlim, [th_order, th_order], linestyle='dotted')

ax[0,0].set_xlim(xlim)
plt.tight_layout()
plt.show()

