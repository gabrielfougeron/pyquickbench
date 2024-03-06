"""
TimeTrains
==========
"""

# %% 
# :class:`pyquickbench.TimeTrain`
# 

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
    
timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')
basename = f'long_bench_1'
timings_filename = os.path.join(timings_folder, basename+'.npz')

# sphinx_gallery_end_ignore

import numpy as np
import pyquickbench
import time



TT = pyquickbench.TimeTrain()

for i in range(10):
    time.sleep(0.0975)
    TT.toc(i)

for i in range(5):
    time.sleep(0.0975)
    TT.toc(i)


print(TT)
d = TT.to_dict()


print(d)
