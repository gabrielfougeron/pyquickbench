"""
Results variability
===================
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
import numpy as np

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)
    
timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')
basename = f'Results_variability_1'
timings_filename = os.path.join(timings_folder, basename+'.npz')

# sphinx_gallery_end_ignore

import pyquickbench
import scipy

rng = np.random.default_rng()
def bar(n):
    if (rng.random() < 0.05):
        return np.nan
    else:
        return n + 4*rng.standard_normal()

all_funs = [
    bar   ,
]

n_bench = 20
all_sizes = np.array(range(n_bench))

pyquickbench.run_benchmark(
    all_sizes               ,
    all_funs                ,
    show = True             ,
# sphinx_gallery_start_ignore
    filename = timings_filename ,
# sphinx_gallery_end_ignore
    mode = 'scalar_output'  ,
    logx_plot = False       ,
    logy_plot = False       ,
) 

# %% 

n_repeat = 10000

# sphinx_gallery_start_ignore
timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')
basename = f'Results_variability_repeat'
timings_filename = os.path.join(timings_folder, basename+'.npz')
# sphinx_gallery_end_ignore

pyquickbench.run_benchmark(
    all_sizes               ,
    all_funs                ,
    n_repeat = n_repeat     ,
    show = True             ,
# sphinx_gallery_start_ignore
    filename = timings_filename ,
# sphinx_gallery_end_ignore
    mode = 'scalar_output'  ,
    logx_plot = False       ,
    logy_plot = False       ,
) 
