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


def test_basic_benchmark(SimpleTimingsBenchmark):
    
    print("Launching basic timings benchmark")

    all_vals = pyquickbench.run_benchmark(
        SimpleTimingsBenchmark.all_args         ,
        SimpleTimingsBenchmark.all_funs         ,
        setup = SimpleTimingsBenchmark.setup    ,
        StopOnExcept = True                     ,
    ) 

def test_all_options_timings(SimpleScalarBenchmark):
    
    all_n_repeat = [1, 2, 3]
    all_nproc = [1, 2]
    all_pooltype = ["phony", "thread", "process"]
    all_ForceBenchmark = [True, False]
    all_PreventBenchmark = [True, False]
    all_StopOnExcept = [True, False]
    all_ShowProgress = [True, False]
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for (
            n_repeat                ,
            nproc                   ,
            pooltype                ,
            ForceBenchmark          ,
            PreventBenchmark        ,
            StopOnExcept            ,
            ShowProgress            ,
        ) in itertools.product(
            all_n_repeat            ,
            all_nproc               ,
            all_pooltype            ,
            all_ForceBenchmark      ,
            all_PreventBenchmark    ,
            all_StopOnExcept        ,
            all_ShowProgress        ,
        ):
        
            print()
            print("Launching benchmark with parameters")
            print(f'n_repeat         = {n_repeat        }')
            print(f'nproc            = {nproc           }')
            print(f'pooltype         = {pooltype        }')
            print(f'ForceBenchmark   = {ForceBenchmark  }')
            print(f'PreventBenchmark = {PreventBenchmark}')
            print(f'StopOnExcept     = {StopOnExcept    }')
            print(f'ShowProgress     = {ShowProgress    }')

            all_vals = pyquickbench.run_benchmark(
                SimpleScalarBenchmark.all_args          ,
                SimpleScalarBenchmark.all_funs          ,
                setup = SimpleScalarBenchmark.setup     ,
                mode                = "scalar_output"   ,
                n_repeat            = n_repeat          ,
                nproc               = nproc             ,
                pooltype            = pooltype          ,
                ForceBenchmark      = ForceBenchmark    ,
                PreventBenchmark    = PreventBenchmark  ,
                StopOnExcept        = StopOnExcept      ,
                ShowProgress        = ShowProgress      ,
            ) 
            
            if PreventBenchmark:
                assert all_vals is None
            else:
                assert isinstance(all_vals, np.ndarray)
                assert all_vals.ndim == 3
                assert all_vals.shape[0] == len(SimpleScalarBenchmark.all_args['n'])
                assert all_vals.shape[1] == len(SimpleScalarBenchmark.all_funs)
                assert all_vals.shape[2] == n_repeat
                



