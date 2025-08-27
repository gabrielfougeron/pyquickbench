import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
import warnings
from test_config import *

import numpy as np
import pyquickbench

def test_factorial_base():

    n = 10
    
    for i in range(1000):
        
        digits = pyquickbench.cython.rankstats.to_factorial_base(i, n)
        ii = pyquickbench.cython.rankstats.from_factorial_base(digits)
        
        assert i == ii
        
        perm = pyquickbench.cython.rankstats.from_left_lehmer(i,n)
        ii == pyquickbench.cython.rankstats.left_lehmer(perm)
        
        assert i == ii

lenlist_list = [
    [100]       ,
    [10]*3      ,
    [2,3,5,7]   ,
]

@pytest.mark.parametrize("lenlist", lenlist_list)
def test_score_to_partial_order_count(lenlist):
    
    nvec = len(lenlist)
    l = [np.random.random(lenlist[ivec]) for ivec in range(nvec)]

    for k in range(1,nvec+1):

        poc_opt = pyquickbench.rankstats.score_to_partial_order_count(k, l)
        poc_bf =  pyquickbench.rankstats.score_to_partial_order_count_brute_force(k, l)
        
        assert np.array_equal(poc_opt, poc_bf)
