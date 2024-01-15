import os
import timeit
import math
import functools
import itertools
import inspect
import warnings
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
import tqdm.notebook

def isnotfinite(arr):
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res

def _mem_shift_restricted(idx, shape, only):
    
    res = 0
    prod_shapes = 1
    
    for i, j in enumerate(only):
        res += idx[i]*prod_shapes
        prod_shapes *= shape[j]
        
    return res

def _prod_rel_shapes(idx_rel, shape):
    
    prod_shapes = 1
    for idx in idx_rel:
        
        prod_shapes *= shape[idx]
    
    return prod_shapes

def _get_rel_idx_from_maze(idx_all_items, idx_vals, shape):

    idx_items = np.zeros(len(idx_all_items), dtype=int)
    for i, j in enumerate(idx_all_items):
        idx_items[i] = idx_vals[j] 

    i_item = _mem_shift_restricted(idx_items, shape, idx_all_items)
    
    return i_item, idx_items

def _in_ipynb():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
   
default_color_list = list(mpl.colors.TABLEAU_COLORS)
default_color_list.append(mpl.colors.BASE_COLORS['b'])
default_color_list.append(mpl.colors.BASE_COLORS['g'])
default_color_list.append(mpl.colors.BASE_COLORS['r'])
default_color_list.append(mpl.colors.BASE_COLORS['m'])
default_color_list.append(mpl.colors.BASE_COLORS['k'])

default_linestyle_list = [
     'solid'                        ,   # Same as (0, ()) or '-'
     'dotted'                       ,   # Same as (0, (1, 1)) or ':'
     'dashed'                       ,   # Same as '--'
     'dashdot'                      ,   # Same as '-.'
     (0, (1, 10))                   ,   # loosely dotted   
     (0, (1, 1))                    ,   # dotted
     (0, (1, 1))                    ,   # densely dotted
     (5, (10, 3))                   ,   # long dash with offset
     (0, (5, 10))                   ,   # loosely dashed
     (0, (5, 5))                    ,   # dashed
     (0, (5, 1))                    ,   # densely dashed
     (0, (3, 10, 1, 10))            ,   # loosely dashdotted
     (0, (3, 5, 1, 5))              ,   # dashdotted
     (0, (3, 1, 1, 1))              ,   # densely dashdotted
     (0, (3, 5, 1, 5, 1, 5))        ,   # dashdotdotted
     (0, (3, 10, 1, 10, 1, 10))     ,   # loosely dashdotdotted
     (0, (3, 1, 1, 1, 1, 1))        ,   # densely dashdotdotted                    
]

default_pointstyle_list = [
    None,   
    "."	, # m00 point
    ","	, # m01 pixel
    "o"	, # m02 circle
    "v"	, # m03 triangle_down
    "^"	, # m04 triangle_up
    "<"	, # m05 triangle_left
    ">"	, # m06 triangle_right
    "1"	, # m07 tri_down
    "2"	, # m08 tri_up
    "3"	, # m09 tri_left
    "4"	, # m10 tri_right
    "8"	, # m11 octagon
    "s"	, # m12 square
    "p"	, # m13 pentagon
    "P"	, # m23 plus (filled)
    "*"	, # m14 star
    "h"	, # m15 hexagon1
    "H"	, # m16 hexagon2
    "+"	, # m17 plus
    "x"	, # m18 x
    "X"	, # m24 x (filled)
    "D"	, # m19 diamond
    "d"	, # m20 thin_diamond
]

def _check_sig_no_pos_only(fun):

    try:
        sig = inspect.signature(fun)
    except ValueError:
        sig = None
        
    if sig is not None:
        for param in sig.parameters.values():
            if param.kind == param.POSITIONAL_ONLY:
                raise ValueError(f'Input argument {param} to provided function {fun.__name__} is positional only. Positional-only arguments are unsupported')
        
def _return_setup_vars_dict(setup, args):

    setup_vars = setup(*args)

    if isinstance(setup_vars, dict):
        setup_vars_dict = setup_vars
    else:
        setup_vars_dict = {setup_vars[i][0] : setup_vars[i][1] for i in range(len(setup_vars))}
        
    return setup_vars_dict

def _load_benchmark_file(filename, all_args_in, shape):
    
    file_base, file_ext = os.path.splitext(filename)
    
    if file_ext == '.npy':
        all_vals = np.load(filename)    

        BenchmarkUpToDate = True
        assert all_vals.ndim == len(shape)
        for loaded_axis_len, expected_axis_len in zip(all_vals.shape, shape.values()):
            BenchmarkUpToDate = BenchmarkUpToDate and (loaded_axis_len == expected_axis_len)

    elif file_ext == '.npz':
        file_content = np.load(filename)

        all_vals = file_content['all_vals']
        
        BenchmarkUpToDate = True
        assert all_vals.ndim == len(shape)
        for loaded_axis_len, expected_axis_len in zip(all_vals.shape, shape.values()):
            BenchmarkUpToDate = BenchmarkUpToDate and (loaded_axis_len == expected_axis_len)
            
        for name, all_args_vals in all_args_in.items():
            assert name in file_content

            for loaded_val, expected_val in zip(file_content[name], all_args_vals):
                assert loaded_val == expected_val
            
    else:
        raise ValueError(f'Unknown file extension {file_ext}')

    # print()
    # print(filename)
    # print(f'{BenchmarkUpToDate = }')

    return all_vals, BenchmarkUpToDate

def _save_benchmark_file(filename, all_vals, all_args):

    file_bas, file_ext = os.path.splitext(filename)
    
    if file_ext == '.npy':
        np.save(filename, all_vals)    
        
    elif file_ext == '.npz':

        np.savez(
            filename                ,
            all_vals = all_vals     ,
            **all_args
        )
    
    else:
        raise ValueError(f'Unknown file extension {file_ext}')
    
def _build_args_shapes(all_args, all_funs, n_repeat):
    
    if not(isinstance(all_args, dict)):
        all_args = {'n': all_args}
    
    assert not('fun' in all_args)
    assert not('repeats' in all_args)
    assert not('all_vals' in all_args)
    
    if isinstance(all_funs, dict):
        all_funs_list = [fun for fun in all_funs.values()]
    else:    
        all_funs_list = [fun for fun in all_funs]
     
    args_shape = {name : len(value) for name, value in all_args.items()}
    
    res_shape = {name : value for name, value in args_shape.items()}
    res_shape['fun'] = len(all_funs)
    res_shape['repeat'] = n_repeat
    
    return all_args, all_funs_list, args_shape, res_shape

class FakeProgressBar(object):
    def __init__(self, *args, **kwargs):
        pass
     
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        pass
    
    def update(self, *args):
        pass
               
def _measure_output(args, setup, all_funs_list, n_repeat, StopOnExcept):

    setup_vars_dict = _return_setup_vars_dict(setup, args)
    n_funs = len(all_funs_list)
    vals = np.full((n_funs, n_repeat), np.nan)
    
    for i_fun, fun in enumerate(all_funs_list):
        
        for i_repeat in range(n_repeat):
            
            try:
                vals[i_fun, i_repeat] = fun(**setup_vars_dict)
            except Exception as exc:
                vals[i_fun, i_repeat] = np.nan
                if StopOnExcept:
                    raise exc
    return vals

def _measure_timings(args, setup, all_funs_list, n_repeat, time_per_test, StopOnExcept):
    
    setup_vars_dict = _return_setup_vars_dict(setup, args)  
    n_funs = len(all_funs_list)
    vals = np.full((n_funs, n_repeat), np.nan)  
    
    for i_fun, fun in enumerate(all_funs_list):
        
        global_dict = {
            'fun'               : fun               ,
            'setup_vars_dict'   : setup_vars_dict   ,
        }

        code = f'fun(**setup_vars_dict)'

        Timer = timeit.Timer(
            code,
            globals = global_dict,
        )

        try:
            # For functions that require caching
            Timer.timeit(number = 1)

            # Estimate time of everything
            n_timeit_0dot2, est_time = Timer.autorange()
            
            if (n_repeat == 1) and (time_per_test == 0.2):
                # Fast track: benchmark is not repeated and autorange results are kept as is.

                vals[i_fun, 0] = est_time / n_timeit_0dot2
                
            else:
                # Full benchmark is run

                n_timeit = math.ceil(n_timeit_0dot2 * time_per_test / est_time / n_repeat)

                times = Timer.repeat(
                    repeat = n_repeat,
                    number = n_timeit,
                )

                vals[i_fun, :] = np.array(times) / n_timeit

        except Exception as exc:

            vals[i_fun, :] = np.nan
            
            if StopOnExcept:
                raise exc
    
    return vals
            
class FakeFuture(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
            
    def add_done_callback(self, callback):
        callback(0)
        
    def result(self):
        return self.fn(*self.args)
        
class PhonyProcessPoolExecutor(object):
    def __init__(self, *args, **kwargs):
        pass
     
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        pass
    
    def submit(self, fn, /, *args):
        return FakeFuture(fn=fn, args=args)
    
AllPoolExecutors = {
    "phony"         :   PhonyProcessPoolExecutor                ,
    "thread"        :   concurrent.futures.ThreadPoolExecutor   ,
    "process"       :   concurrent.futures.ProcessPoolExecutor  ,
}

def default_setup(n):
    return {'n': n}
