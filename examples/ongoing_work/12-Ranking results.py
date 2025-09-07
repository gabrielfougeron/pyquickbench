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
    if ranking[0] == 0:
        return 1
    else:
        return 0    

def metrics(ranking):
    
    return {
        'KendallTauRankCorrelation' : KendallTauRankCorrelation(ranking)    ,
        'FoundExactOrder' : FoundExactOrder(ranking)                        ,
        'FoundFirst' : FoundFirst(ranking)                                  ,
    }

def setup(nvec, nobs, d):
    l = [np.random.random(nobs) + d*ivec for ivec in range(nvec)]
    return {"score_list" : l}

def average_order(score_list):
    nvec = len(score_list)
    moy = np.array([np.mean(score_list[ivec]) for ivec in range(nvec)])
    return np.argsort(moy)
    
def median_order(score_list):
    nvec = len(score_list)
    med = np.array([np.median(score_list[ivec]) for ivec in range(nvec)])
    return np.argsort(med)

def condorcet_order(score_list):
    order_count = pyquickbench.rankstats.score_to_partial_order_count(2, score_list)
    return pyquickbench.rankstats.condorcet_top_order(order_count)

def plackett_luce_order(score_list, k):
    order_count = pyquickbench.rankstats.score_to_partial_order_count(k, score_list)
    A, p, q = pyquickbench.rankstats.build_sinkhorn_problem(order_count)
    u, v = pyquickbench.cython.sinkhorn.sinkhorn_knopp(
        A, p, q,
        numItermax = 10000  ,
        stopThr = 1e-13     ,
    )
    return np.argsort(v)

def count_order_max(score_list):
    nvec = len(score_list)
    order_count = pyquickbench.rankstats.score_to_partial_order_count(nvec, score_list)
    imax = np.argmax(order_count[0,:])
    return pyquickbench.rankstats.from_left_lehmer(imax, nvec)

dmax = 0.1
nd = 32*2-1
do = dmax/nd

nvec = 5

all_args = {
    "nvec"  : [nvec]   ,
    # "nobs"  : [2**i for i in range(10)]  ,
    "nobs"  : [50]  ,
    "d"     : [0. + do*i for i in range(nd+1)]  ,
}

all_funs = [
    average_order       ,
    median_order        ,
    condorcet_order     ,
    count_order_max     ,
]

for k in range(2,nvec+1):
    f = functools.partial(plackett_luce_order, k=k)
    f.__name__ = f'plackett_luce_order_{k}'
    all_funs.append(f)
    
n_repeat = 10

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
    # ForceBenchmark = True                 ,
    deterministic_setup = False         ,
    pooltype = "process"                ,
)

# print(all_values)

plot_intent = {
    "nvec"                      : 'same'            ,
    "nobs"                      : 'same'            ,
    "d"                         : 'points'          ,
    pyquickbench.fun_ax_name    : 'curve_color'     ,
    pyquickbench.out_ax_name    : 'subplot_grid_y'  ,
    pyquickbench.repeat_ax_name : 'reduction_avg'   ,
}


pyquickbench.plot_benchmark(
    all_values                          ,
    all_args                            ,
    all_funs                            ,
    setup = setup                       ,
    wrapup = metrics                    ,
    mode = "vector_output"              ,
    plot_intent = plot_intent           ,
    # plot_type = "bar"                   ,
    # title = "Average number of comparisons needed to sort an array"   ,
    # ylabel = "Kendall Tau Ranking Coefficient"    ,
    xlabel = "Size of the array"        ,
    logx_plot = False                   ,
    logy_plot = False                   ,
    plot_ylim = [-0.05,1.05]                ,
    show = True                         ,
)

pyquickbench.plot_benchmark(
    all_values                          ,
    all_args                            ,
    all_funs                            ,
    setup = setup                       ,
    wrapup = metrics                    ,
    mode = "vector_output"              ,
    plot_intent = plot_intent           ,
    # plot_type = "bar"                   ,
    # title = "Average number of comparisons needed to sort an array"   ,
    ylabel = ""    ,
    # xlabel = "Size of the array"        ,
    logx_plot = False                   ,
    logy_plot = False                   ,
    plot_ylim = [-0.05,len(all_funs)-1 + 0.05]                ,
    transform = 'ascending_rank'        ,
    show = True                         ,
)