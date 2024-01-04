import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
from test_config import *

import numpy as np
import scipy
import pyquickbench



@ProbabilisticTest()
def simple_test():

    def comprehension(n):
        return ['' for _ in range(n)]

    def star_operator(n):
        return ['']*n

    def for_loop_append(n):
        l = []
        for _ in range(n):
            l.append('')
        
    all_funs = [
        comprehension   ,
        star_operator   ,
        for_loop_append ,
    ]
        
    n_bench = 3
    all_sizes = [2**n for n in range(n_bench)]

    pyquickbench.run_benchmark(
        all_sizes   ,
        all_funs    ,
        show = True ,
    ) 




