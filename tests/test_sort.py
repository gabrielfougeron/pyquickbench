import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
from test_config import *

import numpy as np
import pyquickbench

n_list = [10, 100, 1000]

pyquick_bench_sort_list = [
    pyquickbench.sort.merge_sort            ,
    pyquickbench.sort.heap_sort             ,
    pyquickbench.sort.insertion_sort        ,
    pyquickbench.sort.merge_insertion_sort  ,
    pyquickbench.sort.quick_sort            ,
    pyquickbench.sort.binary_insertion_sort ,
]

@pytest.mark.parametrize("n", n_list)
@pytest.mark.parametrize("sort", pyquick_bench_sort_list)
def test_sorts(n, sort):
    
    arr = np.argsort(np.random.random(n))   # integers are easier to manipulate
    sort(arr)

    for i in range(n-1):
        assert i in arr             # making sure no element magically disappeared
        assert arr[i] < arr[i+1]    # making sure array is sorted
        
@pytest.mark.parametrize("n", n_list)
@pytest.mark.parametrize("sort", pyquick_bench_sort_list)
def test_argsorts(n, sort):
    
    arr = np.random.random(n)
    idx = pyquickbench.sort.argsort(arr, sort=sort)

    for i in range(n-1):
        assert i in idx                    # making sure no element magically disappeared
        assert arr[idx[i]] < arr[idx[i+1]] # making sure idx sorts arr
        