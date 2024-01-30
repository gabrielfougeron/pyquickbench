import attrs
import pytest
import inspect
import typing
import warnings
import functools
import numpy as np
import math

@attrs.define
class float_tol:
    atol: float
    rtol: float

@attrs.define
class likelyhood():
    probable        :   float
    not_unlikely    :   float
    uncommon        :   float
    unlikely        :   float
    unbelievable    :   float
    impossible      :   float

def ProbabilisticTest(RepeatOnFail = 10):

    def decorator(test):

        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            
            try:
                return test(*args, **kwargs)

            except AssertionError:
                
                out_str = f"Probabilistic test failed. Running test again {RepeatOnFail} times."
                head_foot = '='*len(out_str)

                print('')
                print(head_foot)
                print(out_str)
                print(head_foot)
                print('')

                for i in range(RepeatOnFail):
                    res = test(*args, **kwargs)

                return res

        return wrapper
    
    return decorator

def RepeatTest(n = 10):

    def decorator(test):

        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            
            for i in range(n):
                res = test(*args, **kwargs)

            return res

        return wrapper
    
    return decorator

@attrs.define
class AllFunBenchmark:
    all_args    :   typing.Dict[str, any]    
    all_funs    :   typing.Dict[str, callable]   
    setup       :   callable

def comprehension(n):
    return [0 for _ in range(n)]
def star_operator(n):
    return [0]*n
def for_loop_append(n):
    l = []
    for _ in range(n):
        l.append(0)
def default_setup(n):
    return {'n': n}
    
@pytest.fixture
def SimpleTimingsBenchmark():
    
    n_bench = 3
    all_sizes = [2**n for n in range(n_bench)]
    
    return AllFunBenchmark(
        all_args = {
            "n" : all_sizes
        },
        all_funs = {
            "comprehension"     : comprehension     ,
            "star_operator"     : star_operator     ,
            "for_loop_append"   : for_loop_append   ,
        },
        setup = default_setup,  
    )   
    
def random_setup(n):
    return {'x': np.random.random((n))}
def numpy_sum(x):
    return np.sum(x)
def math_fsum(x):
    return math.fsum(x)
    
@pytest.fixture
def SimpleScalarBenchmark():
    
    n_bench = 3
    all_sizes = [2**n for n in range(n_bench)]
    
    return AllFunBenchmark(
        all_args = {
            "n" : all_sizes
        },
        all_funs = {
            "numpy_sum"     : numpy_sum ,
            "math_fsum"     : math_fsum ,
        },
        setup = random_setup,  
    )   
    