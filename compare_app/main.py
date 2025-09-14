import sys
import os
import math
import numpy as np

import pyquickbench
import GUI


try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

bench_root = os.path.join(__PROJECT_ROOT__, "AI_bench", "AR")

rank_assign = pyquickbench.ManualRankAssign(bench_root, k=2)



img_compare_GUI = GUI.ImageCompareGUI(rank_assign)
img_compare_GUI()