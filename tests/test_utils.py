import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
import warnings
from test_config import *

import numpy as np
import itertools
import pyquickbench

ndim_list       = [10]
shape_max_list  = [10]
n_only_list     = [5 ]

@RepeatTest()
@pytest.mark.parametrize(("ndim", "shape_max", "n_only"), zip(ndim_list, shape_max_list, n_only_list))
def test_mem_shift(ndim, shape_max, n_only):

    shape = np.random.randint(low=1, high=shape_max+1, size=ndim)
    only = np.random.randint(ndim, size=n_only)
    
    idx = np.array([np.random.randint(shape[only[i]]) for i in range(n_only)])
    
    i_shift = pyquickbench._utils._mem_shift_restricted(idx, only, shape)
    idx_rt  = pyquickbench._utils._from_mem_shift_restricted(i_shift, only, shape)
    
    assert np.array_equal(idx, idx_rt)
    
    n_shift = pyquickbench._utils._prod_rel_shapes(only, shape)
    i_shift = np.random.randint(n_shift)
    
    idx        = pyquickbench._utils._from_mem_shift_restricted(i_shift, only, shape)
    i_shift_rt = pyquickbench._utils._mem_shift_restricted(idx, only, shape)
    
    assert i_shift == i_shift_rt