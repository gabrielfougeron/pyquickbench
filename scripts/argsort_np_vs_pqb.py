
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
basename = 'argsort'
timings_filename = os.path.join(timings_folder, basename+'.npy')

# sphinx_gallery_end_ignore

import numpy as np
import pyquickbench

def setup(n):
    return {'x': np.random.random(n)}

def numpy_argsort(x):
    np.argsort(x)

def pyquickbench_argsort(x):
    pyquickbench.cython.rankstats.insertion_argsort(x)

all_funs = [
    numpy_argsort           ,
    pyquickbench_argsort    ,
]
 
# %% 
# Let's define relevant sizes of lists to be timed.

n_bench = 8
all_sizes = {'n': [2**i for i in range(n_bench)]}
# 
pyquickbench.run_benchmark(
    all_sizes   ,
    all_funs    ,
    setup=setup ,
    show = True ,
    ForceBenchmark=True,
# sphinx_gallery_start_ignore
    filename = timings_filename     ,
# sphinx_gallery_end_ignore
) 
