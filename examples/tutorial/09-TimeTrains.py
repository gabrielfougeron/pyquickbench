"""
TimeTrains
==========
"""

# %% 
# :class:`pyquickbench.TimeTrain`
# 


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
    include_locs = False     ,
    names_reduction = 'sum',
)


for i in range(3):
    time.sleep(0.02)
    TT.toc("toto")

print(TT)
