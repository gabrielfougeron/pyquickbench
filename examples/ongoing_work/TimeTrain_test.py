"""
Time Trains example
===================
"""

# %% 
# This benchmark compares accuracy and efficiency of several summation algorithms in floating point arithmetics

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

import numpy as np
import math as m
import time

import pyquickbench


TT = pyquickbench.TimeTrain(
    # include_locs = True     ,
    names_reduction = 'sum',
)


for i in range(3):
    time.sleep(0.02)
    TT.toc(0)

print(TT)

