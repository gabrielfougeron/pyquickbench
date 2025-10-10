"""
Ranking results
===============
"""

# %% 

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
basename = 'ranking_results_1'
bench_filename = os.path.join(timings_folder, basename+'.npz')

# sphinx_gallery_end_ignore

import numpy as np
import functools
import pyquickbench

def KendallTauRankCorrelation(ranking):
    if np.any(ranking < 0):
        return -1.
    exact_ranking = np.array(range(ranking.shape[0]), dtype=np.intp)
    return pyquickbench.rankstats.KendallTauRankCorrelation(ranking, exact_ranking)
    
def FoundExactOrder(ranking):
    exact_ranking = np.array(range(ranking.shape[0]), dtype=np.intp)
    if np.array_equal(ranking, exact_ranking):
        return 1
    else:
        return 0

def FoundFirst(ranking):
    n = len(ranking)
    if ranking[n-1] == n-1:
        return 1
    else:
        return 0    

def metrics(ranking):
    
    return {
        'KendallTauRankCorrelation' : KendallTauRankCorrelation(ranking)    ,
        'FoundExactOrder' : FoundExactOrder(ranking)                        ,
        'FoundFirst' : FoundFirst(ranking)                                  ,
    }

do = 0.
def setup(nvec, nobs, d, method, nmc, k):
    l = [np.random.random(nobs) + (do+d*ivec) for ivec in range(nvec)]
    return {"score_list" : l, "k":k, "method" : method, "nmc" : nmc}

def plackett_luce_best_count(score_list, k, method, nmc):
    best_count = pyquickbench.rankstats.score_to_partial_best_count(k, score_list, method=method, nmc_all=nmc)
    nvec = len(score_list)
    A = pyquickbench.rankstats.build_sinkhorn_best_count_mat(nvec, k)
    reg_eps = 0.00001
    p, q = pyquickbench.rankstats.build_sinkhorn_rhs_best_count(best_count, reg_eps, nvec)
    
    u, v = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
        A, p, q,
        numItermax = 10000  ,
        stopThr = 1e-13     ,
    )
    return np.argsort(v)

def plackett_luce_best_order(score_list, k, method, nmc):
    order_count_big = pyquickbench.rankstats.score_to_partial_order_count(k, score_list, method=method, nmc_all=nmc)
    order_count_lower = pyquickbench.rankstats.order_count_lower_k(order_count_big, 2)
    nvec = len(score_list)

    reg_eps = 0.00001    
    A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count_lower, reg_eps = reg_eps)
    
    u, v = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
        A, p, q,
        numItermax = 10000  ,
        stopThr = 1e-13     ,
    )
    return np.argsort(v)




all_funs = [
    plackett_luce_best_count    ,
    plackett_luce_best_order    ,
]

# ddmax = 0.4
# nd = 32*4-1
# dd = ddmax/nd
# 
# n_repeat = 10

# all_args = {
#     "nvec"  : [nvec]   ,
#     "nobs"  : [10]  ,
#     "d"     : [dd*i for i in range(nd+1)]  ,
#     "method"  : ["exhaustive"]  ,
#     "nmc"     : [0]  ,
# }

# all_values = pyquickbench.run_benchmark(
#     all_args                            ,
#     all_funs                            ,
#     setup = setup                       ,
#     wrapup = metrics                    ,
#     mode = "vector_output"              ,
#     n_repeat = n_repeat                 ,
#     filename = bench_filename           ,
#     n_out = 3                           ,
#     StopOnExcept = True                 ,
#     ShowProgress = True                 ,
#     deterministic_setup = False         ,
#     # ForceBenchmark = True               ,
#     pooltype = "process"                ,
# )
# 
# plot_intent = {
#     "nvec"                      : 'same'            ,
#     "nobs"                      : 'same'            ,
#     "d"                         : 'points'          ,
#     "method"                    : 'same'            ,
#     "nmc"                       : 'same'            ,
#     pyquickbench.fun_ax_name    : 'curve_color'     ,
#     pyquickbench.out_ax_name    : 'subplot_grid_y'  ,
#     pyquickbench.repeat_ax_name : 'reduction_avg'   ,
# }

# pyquickbench.plot_benchmark(
#     all_values                          ,
#     all_args                            ,
#     all_funs                            ,
#     setup = setup                       ,
#     wrapup = metrics                    ,
#     mode = "vector_output"              ,
#     plot_intent = plot_intent           ,
#     xlabel = "Base distribution separation" ,
#     logx_plot = False                       ,
#     logy_plot = False                       ,
#     plot_ylim = [-0.05,1.05]                ,
#     show = True                             ,
# )

# %%

basename = 'ranking_results_2'
bench_filename = os.path.join(timings_folder, basename+'.npz')

n_repeat = 10000
nvec = 6

nmc_max = 200.
nnmc = 32

all_args = {
    "nvec"  : [nvec]        ,
    "nobs"  : [1000]  ,
    "d"     : [0.1]  ,
    "method"  : ["montecarlo"]  ,
    "nmc"     : [int(nmc_max*i/nnmc) for i in range(1,nnmc+1)]  ,
    "k"     : [2,3,4,5,6],
}

all_values = pyquickbench.run_benchmark(
    all_args                            ,
    all_funs                            ,
    setup = setup                       ,
    wrapup = metrics                    ,
    mode = "vector_output"              ,
    n_repeat = n_repeat                 ,
    filename = bench_filename           ,
    n_out = 3                           ,
    StopOnExcept = True                 ,
    ShowProgress = True                 ,
    deterministic_setup = False         ,
    # ForceBenchmark = True               ,
    pooltype = "process"                ,
)

plot_intent = {
    "nvec"                      : 'same'            ,
    "nobs"                      : 'same'            ,
    "d"                         : 'same'          ,
    "method"                    : 'same'            ,
    "nmc"                       : 'points'            ,
    "k"                         : 'curve_color'            ,
    pyquickbench.fun_ax_name    : 'curve_linestyle'     ,
    pyquickbench.out_ax_name    : 'subplot_grid_y'  ,
    pyquickbench.repeat_ax_name : 'reduction_avg'   ,
}

pyquickbench.plot_benchmark(
    all_values                              ,
    all_args                                ,
    all_funs                                ,
    setup = setup                           ,
    wrapup = metrics                        ,
    mode = "vector_output"                  ,
    plot_intent = plot_intent               ,
    xlabel = "Number of MonteCarlo comparisons" ,
    logx_plot = False                       ,
    logy_plot = False                       ,
    plot_ylim = [-0.05,1.05]                ,
    show = True                             ,
)
