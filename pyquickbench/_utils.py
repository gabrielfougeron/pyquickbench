import os
import timeit
import math
import inspect
import concurrent.futures
import copy
import itertools

import numpy as np

from pyquickbench._defaults import *
from pyquickbench._time_train import TimeTrain

class AutoTimer(timeit.Timer):
    """
    Enhancing Cpython's timeit.Timer.autorange. cf https://github.com/python/cpython/blob/main/Lib/timeit.py
    """

    def autorange(self, callback=None, time_target=0.2):
        """Return the number of loops and time taken so that total time >= time_target.

        Calls the timeit method with increasing numbers from the sequence
        1, 2, 5, 10, 20, 50, ... until the time taken is at least time_target
        second.  Returns (number, time_taken).

        If *callback* is given and is not None, it will be called after
        each trial with two arguments: ``callback(number, time_taken)``.
        """
        i = 1
        while True:
            for j in 1, 2, 5:
                number = i * j
                time_taken = self.timeit(number)
                if callback:
                    callback(number, time_taken)
                if time_taken >= time_target:
                    return (number, time_taken)
            i *= 10

def _count_Truthy(iterator):
    count = 0
    first_true_idx = None
    for i, val in enumerate(iterator):
        if val:
            count+=1
            if first_true_idx is None:
                first_true_idx = i
    return count, first_true_idx

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
        
        BenchmarkUpToDate = (all_vals.ndim == len(shape))
        
        if BenchmarkUpToDate:
        
            for loaded_axis_len, expected_axis_len in zip(all_vals.shape, shape.values()):
                BenchmarkUpToDate = BenchmarkUpToDate and (loaded_axis_len == expected_axis_len)
                
            for name, all_args_vals in all_args_in.items():
                BenchmarkUpToDate = BenchmarkUpToDate and (name in file_content)

                for loaded_val, expected_val in zip(file_content[name], all_args_vals):
                    
                    is_expected = (loaded_val == expected_val)
                    
                    if not(isinstance(is_expected, bool)):
                        is_expected = is_expected.all()
                    
                    BenchmarkUpToDate = BenchmarkUpToDate and is_expected
            
    else:
        raise ValueError(f'Unknown file extension {file_ext}')

    return all_vals, BenchmarkUpToDate

def _save_benchmark_file(filename, all_vals, all_args):
    
    directory = os.path.dirname(filename)
    if not(os.path.isdir(directory)):
        os.makedirs(directory)

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
    
def _build_args_shapes(all_args, all_funs, n_repeat, n_out):
    
    assert not(fun_ax_name in all_args)
    assert not(repeat_ax_name in all_args)
    assert not(out_ax_name in all_args)
    assert not('all_vals' in all_args)
         
    args_shape = {name : len(value) for name, value in all_args.items()}
    
    res_shape = {name : value for name, value in args_shape.items()}
    res_shape[fun_ax_name] = len(all_funs)
    res_shape[repeat_ax_name] = n_repeat
    res_shape[out_ax_name] = n_out
    
    return args_shape, res_shape

def _build_out_names(all_args, setup, all_funs_list):
    
    FoundValidRes = False
    for args in itertools.product(*list(all_args.values())):
        
        setup_vars_dict = _return_setup_vars_dict(setup, args)
        
        for fun in all_funs_list:   
            
            try: # Some structure cannot easily be deepcopied, like those created with Cython.
                setup_vars_dict_cp = copy.deepcopy(setup_vars_dict)
            except TypeError: 
                setup_vars_dict_cp = copy.copy(setup_vars_dict)
                
            try:
                res = fun(**setup_vars_dict_cp)
            except Exception as err:
                pass
            else:
                FoundValidRes = True
                break
                
        if FoundValidRes:
            break
    
    if not(FoundValidRes):
        raise ValueError("Could not find a single valid result in the whole benchmark !")

    if isinstance(res, TimeTrain):
        if res.names_reduction is None:
            res = res.to_dict()
        else:
            res = res.to_dict(names_reduction=all_reductions['sum'])

    if isinstance(res, float):
        n_out = 1
        all_out_names = ["0"]
    elif isinstance(res, dict):
        n_out = len(res)
        all_out_names = list(res.keys())
    else:
        n_out = len(res)
        n_fig = 1 + math.floor(math.log(n_out) / math.log(10))
        all_out_names = [str(i).zfill(n_fig) for i in range(n_out)]
            
    return n_out, all_out_names

class FakeProgressBar(object):
    def __init__(self, *args, **kwargs):
        pass
     
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        pass
    
    def update(self, *args):
        pass
               
def _measure_output(i_args, args, setup, all_funs_list, n_repeat, n_out, StopOnExcept):

    setup_vars_dict = _return_setup_vars_dict(setup, args)
    n_funs = len(all_funs_list)
    vals = np.full((n_funs, n_repeat, n_out), np.nan)
    
    for i_fun, fun in enumerate(all_funs_list):
        
        for i_repeat in range(n_repeat):
            
            try:
                try: # Some structure cannot easily be deepcopied, like those created with Cython.
                    setup_vars_dict_cp = copy.deepcopy(setup_vars_dict)
                except TypeError: 
                    setup_vars_dict_cp = copy.copy(setup_vars_dict)
                    
                res = fun(**setup_vars_dict_cp)
                
                if isinstance(res, TimeTrain):
                    if res.names_reduction is None:
                        res = res.to_dict()
                    else:
                        res = res.to_dict(names_reduction=all_reductions['sum'])
                
                if isinstance(res, float):
                    assert n_out == 1
                    vals[i_fun, i_repeat, :] = res
                    
                elif isinstance(res, np.ndarray):
                    assert res.ndim == 1
                    assert n_out == res.shape[0]
                    vals[i_fun, i_repeat, :] = res
                    
                elif isinstance(res, dict):
                    assert n_out == len(res)
                    for i, val in enumerate(res.values()):
                        vals[i_fun, i_repeat, i] = val

                else:
                    raise TypeError(f'Unknown output type {type(res)}')
                
            except Exception as exc:
                vals[i_fun, i_repeat, :] = np.nan
                if StopOnExcept:
                    raise exc
    return vals

def _measure_timings(i_args, args, setup, all_funs_list, n_repeat, time_per_test, StopOnExcept, WarmUp, all_vals):
    
    setup_vars_dict = _return_setup_vars_dict(setup, args)  
    n_funs = len(all_funs_list)
    vals = np.full((n_funs, n_repeat, 1), np.nan)  
    
    for i_fun, fun in enumerate(all_funs_list):
        
        all_idx_list = list(i_args)
        all_idx_list.append(i_fun)
        all_idx_list.append(0)
        all_idx = tuple(all_idx_list)
        
        DoTimings = not(np.isnan(all_vals[all_idx]))
        
        if DoTimings:
            
            try: # Some structure cannot easily be deepcopied, like those created with Cython.
                setup_vars_dict_cp = copy.deepcopy(setup_vars_dict)
            except TypeError: 
                setup_vars_dict_cp = copy.copy(setup_vars_dict)
            
            global_dict = {
                fun_ax_name         : fun               ,
                'setup_vars_dict'   : setup_vars_dict_cp,
            }

            code = f'{fun_ax_name}(**setup_vars_dict)'

            Timer = AutoTimer(
                code,
                globals = global_dict,
            )

            try:
                
                if WarmUp:
                    Timer.timeit(number = 1)

                # Estimate time of everything
                n_timeit_autorange, est_time = Timer.autorange(time_target = time_per_test)
                
                if (n_repeat == 1):
                    # Fast track: benchmark is not repeated and autorange results are kept as is.

                    vals[i_fun, 0, 0] = est_time / n_timeit_autorange
                    
                else:
                    # Full benchmark is run

                    n_timeit = math.ceil(n_timeit_autorange * time_per_test / est_time / n_repeat)

                    times = Timer.repeat(
                        repeat = n_repeat,
                        number = n_timeit,
                    )
                    
                    times_arr = np.array(times) / n_timeit

                    vals[i_fun, :, 0] = times_arr

            except Exception as exc:

                vals[i_fun, :, 0] = np.nan
                
                if StopOnExcept:
                    raise exc
    
    return vals
            
class FakeFuture(object):
    def __init__(self, fn, *args):
        self.res = fn(*args)
            
    def add_done_callback(self, callback):
        callback(self)
        
    def result(self):
        return self.res
        
class PhonyProcessPoolExecutor(object):
    def __init__(self, *args, **kwargs):
        pass
     
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        pass
    
    def submit(self, fn, *args):
        return FakeFuture(fn, *args)

AllPoolExecutors = {
    "phony"         :   PhonyProcessPoolExecutor                ,
    "thread"        :   concurrent.futures.ThreadPoolExecutor   ,
    "process"       :   concurrent.futures.ProcessPoolExecutor  ,
}

def _values_reduction(all_vals, idx_vals, idx_points, idx_all_reduction):

    idx_vals = idx_vals.copy()
    idx_vals[idx_points] = slice(None)

    idx_to_reduction = {idx_points:"points"}
    
    for key, idx in idx_all_reduction.items():
        for i in idx:
            if idx_vals[i] is None:
                idx_vals[i] = slice(None)
                idx_to_reduction[i] = key
                
    idx_to_reduction = dict(sorted(idx_to_reduction.items()))
    
    idx_vals_tuple = tuple(idx_vals)

    reduced_val = all_vals[idx_vals_tuple].copy()

    for red_idx_rel, (red_idx_abs, red_name) in enumerate(idx_to_reduction.items()):

        if red_name != "points":
            reduced_val = all_reductions[red_name](reduced_val, axis=red_idx_rel, keepdims=True)
    
    return reduced_val

def _build_product_legend(idx_curve, plot_legend_curve, name_curve, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, label):

    for idx, plot_legend, cat_name in zip(idx_curve, plot_legend_curve, name_curve):

        if plot_legend:
            
            curve_vals = all_args.get(cat_name)
            if curve_vals is None:
                if cat_name == fun_ax_name:
                    val_name = all_fun_names_list[idx]
                elif cat_name == repeat_ax_name:
                    val_name = str(idx+1)
                elif cat_name == out_ax_name:
                    val_name = all_out_names_list[idx]
                else:
                    raise ValueError("Could not figure out name")
            else:
                val_name = str(curve_vals[idx])
            
            if KeyValLegend:
                label += f'{cat_name} : {val_name}, '
            else:
                label += f'{val_name}, '
            
    return label

def _choose_idx_val(name, all_idx, all_val, all_args, all_fun_names_list):
    
    if isinstance(all_idx, dict):
        idx = all_idx.get(name)
    elif isinstance(all_val, dict):
        val = all_val.get(name)
        
        if val is None:
            idx = None
        else:
        
            if name == fun_ax_name:
                idx = all_fun_names_list.index(val)
            elif name == repeat_ax_name:
                idx = int(val)
            else:
                search_in = all_args[name]
                if isinstance(search_in, list):
                    idx = search_in.index(val)
                elif isinstance(search_in, np.ndarray):
                    idx = np.nonzero(search_in == val)[0]
                else:
                    raise NotImplementedError(f"Searching in {type(all_args[name])} not yet supported.")
    else:
        idx = None
        
    return idx

def _arg_names_list_to_idx(name_list, all_args):
    
    idx_list = []
    
    for name in name_list:
        
        found = False
        for idx, arg_name in enumerate(all_args):
            if (name == arg_name):
                found = True
                break
            
        if found:
            idx_list.append(idx)
        else:
            raise ValueError(f"Declared monotonic axis {name} is not the name of an argument.")
    
    return idx_list

def _treat_future_result(future):
    
    all_idx_list = list(future.i_args)
    all_idx_list.append(slice(None))
    all_idx_list.append(slice(None))
    all_idx = tuple(all_idx_list)
    
    try:
        
        res = future.result()
        
        for i_fun in range(res.shape[0]):
            
            all_idx_list = list(future.i_args)
            
            tot_time = np.sum(res[i_fun,:])
            if (tot_time > future.timeout):
                
                for idx in future.MonotonicAxes_idx:
                    
                    all_idx_list[idx] = slice(future.i_args[idx], None, None)
        
            all_idx_list.append(i_fun)
            all_idx_list.append(slice(None))
            all_idx = tuple(all_idx_list)
            
            future.all_vals[all_idx] = np.nan
        
        all_idx_list = list(future.i_args)
        all_idx_list.append(slice(None))
        all_idx_list.append(slice(None))
        all_idx = tuple(all_idx_list)
        future.all_vals[all_idx] = res

    except Exception as exc:
        future.all_vals[all_idx] = np.nan
        if future.StopOnExcept:
            raise exc
        
def _product(it):
    res = 1
    for item in it:
        res *= item
        
    return res