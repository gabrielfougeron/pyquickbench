"""
Time Trains
===========
"""

# %% 
# As demonstrated in earlier posts in the tutorial, :mod:`pyquickbench` can be useful to measure the wall time of python functions. More often than not however, it can be useful to have a more precise idea of where CPU cycles are spent. This is the raison d'être of :class:`pyquickbench.TimeTrain`. As shown in the following few lines, using a :class:`pyquickbench.TimeTrain` is extremely simple: simply call the :meth:`pyquickbench.TimeTrain.toc` method between snippets of code you want to time and :mod:`pyquickbench` takes care of the rest!

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
basename = 'TimeTrain'
timings_filename = os.path.join(timings_folder, basename+'.npy')
    
import numpy as np
import math as m
import functools

# sphinx_gallery_end_ignore

import time
import pyquickbench

# %% 

TT = pyquickbench.TimeTrain()

time.sleep(0.01)
TT.toc()

time.sleep(0.02)
TT.toc()

time.sleep(0.03)
TT.toc()

time.sleep(0.04)
TT.toc()

time.sleep(0.01)
TT.toc()    

print(TT)


# %% 
# Individual calls to :meth:`pyquickbench.TimeTrain.toc` can be named.

TT = pyquickbench.TimeTrain()

for i in range(3):
    time.sleep(0.01)
    TT.toc("repeated")

for i in range(3):
    time.sleep(0.01)
    TT.toc(f"unique {i+1}")

print(TT)

# %% 
# Timing measurements relative to identical names can be reduced using any reduction method in :data:`pyquickbench.all_reductions`.

TT = pyquickbench.TimeTrain(
    names_reduction = 'sum',
)

for i in range(3):
    time.sleep(0.01)
    TT.toc("repeated")

for i in range(3):
    time.sleep(0.01)
    TT.toc(f"unique {i+1}")

print(TT)

# %% 
# Relative measured times can be reported using the keyword ``relative_timings``.

TT = pyquickbench.TimeTrain(
    names_reduction = 'avg',
    relative_timings = True,
)

for i in range(3):
    time.sleep(0.01)
    TT.toc("repeated")

for i in range(3):
    time.sleep(0.01)
    TT.toc(f"unique {i+1}")

print(TT)


# %% 
# Reductions make locations ill-defined, which is why :class:`pyquickbench.TimeTrain` is issuing a warning. Another good reason to disable location recording is that the corresponding call to :func:`python:inspect.stack` can be non-negligible (around 0.01s on a generic laptop computer).
# Displaying locations can be disabled like so:

TT = pyquickbench.TimeTrain(
    names_reduction = 'sum',
    include_locs = False,
)

for i in range(3):
    time.sleep(0.01)
    TT.toc("repeated")

for i in range(3):
    time.sleep(0.01)
    TT.toc(f"unique {i+1}")

print(TT)


# %% 
# TimeTrains can also time calls to a function. The function :meth:`pyquickbench.TimeTrain.tictoc` will instrument a given function to record its execution time. The most straightforward is to use :meth:`pyquickbench.TimeTrain.tictoc` with a decorator syntax:

TT = pyquickbench.TimeTrain()

@TT.tictoc
def wait_a_bit():
    time.sleep(0.01)

@TT.tictoc
def wait_more():
    time.sleep(0.01)
    

wait_a_bit()
wait_more()

print(TT)

# %% 
# Using :meth:`pyquickbench.TimeTrain.tictoc` with a regular wrapping syntax might lead to surprizing results:
    
TT = pyquickbench.TimeTrain()
    
def wait_unrecorded():
    time.sleep(0.01)
    
wait_recorded = TT.tictoc(wait_unrecorded)

wait_unrecorded()
wait_recorded()

print(TT)

# %% 
# This behavior is to be expected because under the hood, :meth:`pyquickbench.TimeTrain.tictoc` uses the ``__name__`` attribute of the wrapped function. 
    
print(f'{wait_recorded.__name__ = }')

# %% 
# Overriding the ``__name__`` of the wrapped function gives the expected result:

TT = pyquickbench.TimeTrain()
    
def wait_unrecorded():
    time.sleep(0.01)
    
wait_recorded = TT.tictoc(wait_unrecorded)
wait_recorded.__name__ = 'wait_recorded'

wait_unrecorded()
wait_recorded()

print(TT)
# %% 
# More simply, you can set a custom name with the following syntax:

TT = pyquickbench.TimeTrain()
    
def wait_unrecorded():
    time.sleep(0.01)
    
wait_recorded = TT.tictoc(wait_unrecorded, name = "my_custom_name")

wait_unrecorded()
wait_recorded()

print(TT)

# %% 
# By default, a :class:`pyquickbench.TimeTrain` will not show timings occuring in between decorated function. This behavior can be overriden setting the ``ignore_names`` argument to an empty iterator:

TT = pyquickbench.TimeTrain(ignore_names = [])
    
def wait_unrecorded():
    time.sleep(0.01)
    
@TT.tictoc
def wait_recorded():
    time.sleep(0.02)

wait_unrecorded()
wait_recorded()

print(TT)


# %% 
# Let's revisit the benchmark in :ref:`sphx_glr__build_auto_examples_tutorial_09-Vector_output.py` and measure the execution time of different parts of the function ``uniform_quantiles`` using :class:`pyquickbench.TimeTrain`.
# 

def uniform_quantiles(n, m):
   
    TT = pyquickbench.TimeTrain(
        include_locs = False,
    )
    
    vec = np.random.random((n+1))
    TT.toc("Random sampling")
    
    vec.sort()
    TT.toc("Sorting")
    
    res = np.array([abs(vec[(n // m)*i]) for i in range(m+1)])
    TT.toc("Building result")

    return TT

# %% 
# 
# This function can be divided up into three main parts:
# 
# * A random sampling phase, where data is generated. This part is expected to scale as :math:`\mathcal{O}(n)`.
# * A sorting phase where the random vector is sorted in-place. This part is expected to scale as :math:`\mathcal{O}(n\log(n))`, and thus be dominant for large :math:`n`.
# * A last phase where estimated quantiles are built and returned. This phase is expected to scale as :math:`\mathcal{O}(1)` and be negligible for large :math:`n`.

    
m = 10
uniform_decile = functools.partial(uniform_quantiles, m=m)
uniform_decile.__name__ = "uniform_decile"
    
all_funs = [
    uniform_decile   ,   
]

n_bench = 20
all_sizes = [m * 2**n for n in range(n_bench)]

n_repeat = 100
    
plot_intent = {
    pyquickbench.default_ax_name : "points"         ,   
    pyquickbench.out_ax_name : "curve_color"        ,   
}

pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    n_repeat = n_repeat             ,
    mode = "vector_output"          ,
    StopOnExcept = True             ,
    plot_intent = plot_intent       ,
    show = True                     ,
# sphinx_gallery_start_ignore
    filename = timings_filename     ,
# sphinx_gallery_end_ignore
) 

