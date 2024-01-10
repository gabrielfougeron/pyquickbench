import os
import timeit
import math
import itertools
import typing
import inspect

import numpy as np
import numpy.typing
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure
import tqdm
import tqdm.notebook

def isnotfinite(arr):
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res

def _mem_shift(idx, shape):
    
    res = 0
    prod_shapes = 1
    
    # for i in reversed(range(len(shape))): # less efficient : https://stackoverflow.com/questions/7286365/print-a-list-in-reverse-order-with-range/44519681#44519681
    for i in range(len(shape)-1,-1,-1):
        res += idx[i]*prod_shapes
        prod_shapes *= shape[i]
        
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
    #     
    # print()
    # print(idx_items, type(idx_items), idx_items.dtype)
    # print(idx_all_items, type(idx_all_items), idx_all_items.dtype)
    # print(shape, type(shape))
    # print(shape[idx_all_items])
              
    i_item = _mem_shift_restricted(idx_items, shape, idx_all_items)
    
    return i_item, idx_items

# def _in_ipynb():
#     try:
#         cfg = get_ipython().config 
#         if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
#             return True
#         else:
#             return False
#     except NameError:
#         return False
#     
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

def _return_setup_vars_dict(setup, args, fun):
    
    try:
        sig = inspect.signature(fun)
    except ValueError:
        sig = None
        
    if sig is not None:
        for param in sig.parameters.values():
            if param.kind == param.POSITIONAL_ONLY:
                raise ValueError(f'Input argument {param} to provided function {fun.__name__} is positional only. Positional-only arguments are unsupported')
        
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

def run_benchmark(
    all_args        : dict | typing.Iterable                                ,
    all_funs        : dict | typing.Iterable                                ,
    mode            : str           = "timings"                             ,
    setup           : typing.Callable[[int], typing.Dict[str, typing.Any]]
                                    = (lambda n: {'n': n})                  ,
    n_repeat        : int           = 1                                     ,
    time_per_test   : float         = 0.2                                   ,
    filename        : str | None    = None                                  ,
    ForceBenchmark  : bool          = False                                 ,
    show            : bool          = False                                 ,
    StopOnExcept    : bool          = False                                 ,
    ShowProgress    : bool          = False                                 ,
    **show_kwargs   : typing.Dict[str, typing.Any]                          ,
) -> np.typing.NDArray[np.float64] | None :
    
    if filename is None:
        Load_timings_file = False
        Save_timings_file = False
        
    else:
        Load_timings_file =  os.path.isfile(filename) and not(ForceBenchmark)
        Save_timings_file = True

    all_args, all_funs_list, args_shape, res_shape = _build_args_shapes(all_args, all_funs, n_repeat)

    if Load_timings_file:
        
        try:
            
            all_vals, BenchmarkUpToDate = _load_benchmark_file(filename, all_args, res_shape)

            DoBenchmark = not(BenchmarkUpToDate)

        except Exception as exc:
            
            if StopOnExcept:
                raise exc
            
            DoBenchmark = True
            
    else:

        DoBenchmark = True

    if DoBenchmark:

        all_vals = np.full(list(res_shape.values()), np.nan)

        if mode == "timings":
            
            benchmark_iterator = zip(
                itertools.product(*[range(i) for i in args_shape.values()]) ,
                itertools.product(*list(all_args.values()))                 ,
            )

            if ShowProgress:
                
                if (_in_ipynb()):
                    
                    benchmark_iterator = tqdm.notebook.tqdm(
                        iterable = benchmark_iterator,
                        total = math.prod(args_shape.values())
                    )
                    
                else:
                    
                    benchmark_iterator = tqdm.tqdm(
                        iterable = benchmark_iterator,
                        total = math.prod(args_shape.values())
                    )
            
            for i_args, args in benchmark_iterator:

                for i_fun, fun in enumerate(all_funs_list):

                    setup_vars_dict = _return_setup_vars_dict(setup, args, fun)
                    
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

                            all_idx = list(i_args)
                            all_idx.append(i_fun)
                            all_idx.append(0)
                            all_idx_tuple = tuple(all_idx)
                            all_vals[all_idx_tuple] = est_time / n_timeit_0dot2
                            
                        else:
                            # Full benchmark is run

                            n_timeit = math.ceil(n_timeit_0dot2 * time_per_test / est_time / n_repeat)

                            times = Timer.repeat(
                                repeat = n_repeat,
                                number = n_timeit,
                            )
                            all_idx = list(i_args)
                            all_idx.append(i_fun)
                            all_idx.append(slice(None))
                            all_idx_tuple = tuple(all_idx)
                            all_vals[all_idx_tuple] = np.array(times) / n_timeit

                    except Exception as exc:
                        if StopOnExcept:
                            raise exc
                        
                        all_idx = list(i_args)
                        all_idx.append(i_fun)
                        all_idx.append(slice(None))
                        all_idx_tuple = tuple(all_idx)
                        all_vals[all_idx_tuple] = np.nan
                        
        elif mode == "scalar_output":    
            
            for i_args, args in zip(
                itertools.product(*[range(i) for i in args_shape.values()])   ,
                itertools.product(*list(all_args.values()))     ,
            ):

                for i_fun, fun in enumerate(all_funs_list):
                    
                    setup_vars_dict = _return_setup_vars_dict(setup, args, fun)
                    
                    for i_repeat in range(n_repeat):
                        
                        try:

                            out_val = fun(**setup_vars_dict)
                        
                        except Exception as exc:
                            if StopOnExcept:
                                raise exc

                            out_val = np.nan
                            
                        all_idx = list(i_args)
                        all_idx.append(i_fun)
                        all_idx.append(i_repeat)
                        all_idx_tuple = tuple(all_idx)
                        
                        all_vals[all_idx_tuple] = out_val
                        
        else:
            
            raise ValueError(f'Unknown mode {mode}')

        # Sort values along "repeat" axis, taking care of nans.
        idx = np.argsort(all_vals, axis=-1)
        all_vals = np.take_along_axis(all_vals, idx, axis=-1)
        
        if Save_timings_file:
            _save_benchmark_file(filename, all_vals, all_args)
            
    # print(f'{all_vals = }')
            
    if show:
        return plot_benchmark(
            all_vals            ,
            all_args            ,
            all_funs            ,
            mode = mode         ,
            show = show         ,
            **show_kwargs       ,
        )

    return all_vals

all_plot_intents = [
    'single_value'      ,
    'points'            ,
    'same'              ,
    'curve_color'       ,
    'curve_linestyle'   ,
    'curve_pointstyle'  ,
    'subplot_grid_x'    ,
    'subplot_grid_y'    ,
    'reduction_avg'     ,
]

def plot_benchmark(
    all_vals                : np.typing.ArrayLike   ,
    all_args                : dict | typing.Iterable                                        ,
    all_funs                : typing.Dict[str, callable] |
                              typing.Iterable[str] | 
                              None                              = None                      ,
    all_fun_names           : typing.Iterable[str] | None       = None                      ,
    plot_intent         : typing.Iterable[str] | None       = None                      ,        
    mode                    : str                               = "timings"                 ,
    all_xvalues             : np.typing.ArrayLike | None        = None                      ,
    # all_x_scalings          : np.typing.ArrayLike | None        = None                  ,
    # all_y_scalings          : np.typing.ArrayLike | None        = None                  ,
    color_list              : list                              = default_color_list        ,
    linestyle_list          : list                              = default_linestyle_list    ,
    pointstyle_list         : list                              = default_pointstyle_list   ,
    single_values_idx       : dict | None                       = None                      ,         
    logx_plot               : bool | None                       = None                      ,
    logy_plot               : bool | None                       = None                      ,
    plot_ylim               : tuple | None                      = None                      ,
    plot_xlim               : tuple | None                      = None                      ,
    clip_vals               : bool                              = False                     ,
    stop_after_first_clip   : bool                              = False                     ,
    show                    : bool                              = False                     ,
    fig                     : matplotlib.figure.Figure | None   = None                      ,
    ax                      : plt.Axes | None                   = None                      ,
    dpi                     : int                               = 150                       ,
    pxl_per_plot_x          : int                               = 1600                      ,
    pxl_per_plot_y          : int                               = 800                       ,
    sharex                  : bool                              = True                      ,
    sharey                  : bool                              = False                     ,
    title                   : str | None                        = None                      ,
    xlabel                  : str | None                        = None                      ,
    ylabel                  : str | None                        = None                      ,
    plot_legend             : bool                              = True                      ,
    legend_location         : str                               = 'upper left'              ,
    plot_grid               : bool                              = True                      ,
    transform               : str | None                        = None                      ,
    alpha                   : float                             = 1.                        ,
    relative_to             : np.typing.ArrayLike | None        = None                      ,
) -> None :

    # TODO: Change that
    ProductLegend = True
    
    # print(f'{all_vals = }')
    
    all_args, _, args_shape, res_shape = _build_args_shapes(all_args, all_funs, all_vals.shape[-1])

    assert isinstance(all_vals, np.ndarray)
    assert all_vals.ndim == len(res_shape)
    for loaded_axis_len, expected_axis_len in zip(all_vals.shape, res_shape.values()):
        assert loaded_axis_len == expected_axis_len

    n_funs = res_shape['fun']
    n_repeat = res_shape['repeat']

    if all_fun_names is None:
        
        if all_funs is None:
            all_fun_names_list = ['Anonymous function']*n_funs

        else:
            
            all_fun_names_list = []
            
            if isinstance(all_funs, dict):
                for name, fun in all_funs.items():
                    all_fun_names_list.append(name)
            
            else:    
                for fun in all_funs:
                    all_fun_names_list.append(getattr(fun, '__name__', 'Anonymous function'))
                    
    else:
        all_fun_names_list = [name for name in all_fun_names]
    
    assert n_funs == len(all_fun_names_list)

    if plot_intent is None:
        
        plot_intent = {name: 'points' if (i==0) else 'curve_color' for i, name in enumerate(all_args)}
        plot_intent['fun'] = 'curve_color'
        plot_intent['repeat'] = 'same'

    else:
        
        assert isinstance(plot_intent, dict)
        
        if 'fun' not in plot_intent:
            plot_intent['fun'] = 'curve_color'
        if 'repeat' not in plot_intent:
            plot_intent['repeat'] = 'same'
        
        assert len(plot_intent) == all_vals.ndim
        for name, intent in plot_intent.items():
            if not(name in res_shape):
                raise ValueError(f'Unknown argument {name} in plot_intent')
            
            if not(intent in all_plot_intents):
                raise ValueError(f'Unknown intent {intent} in plot_intent. Possible values are: {all_plot_intents}')
    
    n_points = 0
    idx_points = -1

    idx_all_same = []
    idx_all_single_value = []
    idx_all_curve_color = []
    idx_all_curve_linestyle = []
    idx_all_curve_pointstyle = []
    idx_all_subplot_grid_x = []
    idx_all_subplot_grid_y = []
    idx_all_reduction_avg = []
    
    idx_single_value = []
    name_curve_color = []
    name_curve_linestyle = []
    name_curve_pointstyle = []
    name_subplot_grid_x = []
    name_subplot_grid_y = []
    
    for i, (name, value) in enumerate(plot_intent.items()):
        
        if value == 'points':
            n_points += 1
            idx_points = i
            name_points = name
        elif value == 'same':
            idx_all_same.append(i)
        elif value == 'single_value':
            if single_values_idx is None:
                raise ValueError("Argument single_values_idx was not given. It is required to fix a value of an argument in the benchmark.")
            
            idx_all_single_value.append(i)
            idx_single_value.append(single_values_idx[name])            
            
        elif value == 'curve_color':
            idx_all_curve_color.append(i)
            name_curve_color.append(name)
        elif value == 'curve_linestyle':
            idx_all_curve_linestyle.append(i)
            name_curve_linestyle.append(name)
        elif value == 'curve_pointstyle':
            idx_all_curve_pointstyle.append(i)
            name_curve_pointstyle.append(name)
        elif value == 'subplot_grid_x':
            idx_all_subplot_grid_x.append(i)
            name_subplot_grid_x.append(name)
        elif value == 'subplot_grid_y':
            idx_all_subplot_grid_y.append(i)
            name_subplot_grid_y.append(name)
        elif value == 'reduction_avg':
            idx_all_reduction_avg.append(i)
            
    if (n_points != 1):
        raise ValueError(f"There should be exactly one plot_intent named 'points'. There are currently {n_points}.")
    
    idx_all_curves = []
    idx_all_curves.extend(idx_all_same)
    idx_all_curves.extend(idx_all_curve_color)
    idx_all_curves.extend(idx_all_curve_linestyle)
    idx_all_curves.extend(idx_all_curve_pointstyle)
    idx_all_curves.extend(idx_all_subplot_grid_x)
    idx_all_curves.extend(idx_all_subplot_grid_y)
    
    idx_all_same                = np.array(idx_all_same             )
    idx_all_single_value        = np.array(idx_all_single_value     )
    idx_all_curve_color         = np.array(idx_all_curve_color      )
    idx_all_curve_linestyle     = np.array(idx_all_curve_linestyle  )
    idx_all_curve_pointstyle    = np.array(idx_all_curve_pointstyle )
    idx_all_subplot_grid_x      = np.array(idx_all_subplot_grid_x   )
    idx_all_subplot_grid_y      = np.array(idx_all_subplot_grid_y   )
    idx_all_reduction_avg       = np.array(idx_all_reduction_avg    )
    
    n_subplot_grid_x = _prod_rel_shapes(idx_all_subplot_grid_x, all_vals.shape)
    n_subplot_grid_y = _prod_rel_shapes(idx_all_subplot_grid_y, all_vals.shape)

    if logx_plot is None:
        logx_plot = (transform is None)
        
    if logy_plot is None:
        logy_plot = (transform is None)
    
    # TODO: assess whether those need to stay
# 
#     if all_x_scalings is None:
#         all_x_scalings = np.ones(n_funs)
#     else:
#         assert all_x_scalings.shape == (n_funs,)
# 
#     if all_y_scalings is None:
#         all_y_scalings = np.ones(n_funs)
#     else:
#         assert all_y_scalings.shape == (n_funs,)

    if (ax is None) or (fig is None):

        figsize = (
            n_subplot_grid_x * pxl_per_plot_x/dpi,
            n_subplot_grid_y * pxl_per_plot_y/dpi,
        )

        fig, ax = plt.subplots(
            nrows = n_subplot_grid_y    ,
            ncols = n_subplot_grid_x    ,
            sharex = sharex             ,
            sharey = sharey             ,
            figsize = figsize           ,
            dpi = dpi                   ,
            squeeze = False             ,
        )
        
    else:
        
        if isinstance(ax, np.ndarray):
            
            assert ax.shape[0] == n_subplot_grid_y
            assert ax.shape[1] == n_subplot_grid_x
            
        elif isinstance(ax, plt.Axes):
            
            assert n_subplot_grid_x == 1
            assert n_subplot_grid_y == 1
            
            ax = np.array([[ax]])
        
        else:
            raise TypeError(f'Non compatible type for argument "ax": {type(ax)}')
        
    # TODO : Adapt this
    # if (relative_to is None):
    #     relative_to_array = np.ones(list(args_shape.values()))
    #     
    # else:
    #     raise NotImplementedError
    #     if isinstance(relative_to, np.ndarray):
    #         relative_to_array = relative_to
    #         assert relative_to_array.shape == tuple(args_shape.values())
    #     
    #     else:
    #         
    #         if isinstance(relative_to, int):
    #             relative_to_idx = relative_to
    #             
    #         elif isinstance(relative_to, str):
    #             try:
    #                 relative_to_idx = all_names_list.index(relative_to)
    #             except ValueError:
    #                 raise ValueError(f'{relative_to} is not a known name')
    #         else:
    #             raise ValueError(f'Invalid relative_to argument {relative_to}')
    #         
    #         relative_to_array = (np.sum(all_vals[:, relative_to_idx, :], axis=1) / n_repeat)
        
        
    n_colors = len(color_list)
    n_linestyle = len(linestyle_list)
    n_pointstyle = len(pointstyle_list)

    if clip_vals and (plot_ylim is None):
        raise ValueError('Need a range to clip values')

    leg_patch = [[[] for _ in range(n_subplot_grid_y)] for __ in range(n_subplot_grid_x)]
    
    for idx_curve in itertools.product(*[range(all_vals.shape[i]) for i in idx_all_curves]):
        
        idx_vals = [-1] * all_vals.ndim
        for i, j in zip(idx_curve, idx_all_curves):
            idx_vals[j] = i
            
        for i, j in zip(idx_single_value, idx_all_single_value):
            idx_vals[j] = i
        
        i_color, idx_curve_color = _get_rel_idx_from_maze(idx_all_curve_color, idx_vals, all_vals.shape)
        i_linestyle, idx_curve_linestyle = _get_rel_idx_from_maze(idx_all_curve_linestyle, idx_vals, all_vals.shape)
        i_pointstyle, idx_curve_pointstyle = _get_rel_idx_from_maze(idx_all_curve_pointstyle, idx_vals, all_vals.shape)
        i_subplot_grid_x, idx_subplot_grid_x =  _get_rel_idx_from_maze(idx_all_subplot_grid_x, idx_vals, all_vals.shape)
        i_subplot_grid_y, idx_subplot_grid_y =  _get_rel_idx_from_maze(idx_all_subplot_grid_y, idx_vals, all_vals.shape)
        i_same, idx_same =  _get_rel_idx_from_maze(idx_all_same, idx_vals, all_vals.shape)
        
        color = color_list[i_color % n_colors]
        linestyle = linestyle_list[i_linestyle % n_linestyle]
        pointstyle = pointstyle_list[i_pointstyle % n_pointstyle]
        cur_ax = ax[i_subplot_grid_y, i_subplot_grid_x]
        OnlyThisOnce = (i_same == 0)

        if OnlyThisOnce:
            
            if ProductLegend: 
                
                ncats = len(idx_curve_color) + len(idx_curve_linestyle) + len(idx_curve_pointstyle)
                KeyValLegend = (ncats > 1)
            
                label = ''
                for i, idx in enumerate(idx_curve_color):

                    cat_name = name_curve_color[i]
                    curve_vals = all_args.get(cat_name)
                    if curve_vals is None:
                        if cat_name == 'fun':
                            val_name = all_fun_names_list[idx]
                        else:
                            raise ValueError("Could not figure out name")
                    else:
                        val_name = str(curve_vals[idx])
                    
                    if KeyValLegend:
                        label += f'{cat_name} : {val_name}, '
                    else:
                        label += f'{val_name}, '

                for i, idx in enumerate(idx_curve_linestyle):
                    
                    cat_name = name_curve_linestyle[i]
                    curve_vals = all_args.get(cat_name)
                    if curve_vals is None:
                        if cat_name == 'fun':
                            val_name = all_fun_names_list[idx]
                        else:
                            raise ValueError("Could not figure out name")
                    else:
                        val_name = str(curve_vals[idx])
                    
                    if KeyValLegend:
                        label += f'{cat_name}: {val_name}, '
                    else:
                        label += f'{val_name}, '        
                    
                for i, idx in enumerate(idx_curve_pointstyle):
                    
                    cat_name = name_curve_pointstyle[i]
                    curve_vals = all_args.get(cat_name)
                    if curve_vals is None:
                        if cat_name == 'fun':
                            val_name = all_fun_names_list[idx]
                        else:
                            raise ValueError("Could not figure out name")
                    else:
                        val_name = str(curve_vals[idx])
                    
                    if KeyValLegend:
                        label += f'{cat_name} : {val_name}, '
                    else:
                        label += f'{val_name}, '   
                
                label = label[:-2] # Removing final comma
                
                leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                    mpl.lines.Line2D([], []     ,
                        color = color           ,
                        label = label           ,
                        linestyle = linestyle   ,
                        marker = pointstyle     ,
                        markersize = 10         ,
                    )
                )

        idx_vals[idx_points] = slice(None)
        idx_vals_tuple = tuple(idx_vals)
        
        if all_xvalues is None:
            plot_x_val = all_args[name_points]
        else:   
            plot_x_val = all_xvalues[idx_vals_tuple]
            
        plot_y_val = all_vals[idx_vals_tuple]
        
        # print()
        # print(plot_x_val)
        # print(plot_y_val)
        # print(color)
        # print(linestyle)
        # print(pointstyle)
        
        cur_ax.plot(
            plot_x_val              ,
            plot_y_val              ,
            color = color           ,
            linestyle = linestyle   ,
            marker = pointstyle     ,
        )
        
        
        
        
        
        
        
        
# 
#         for i_repeat in range(n_repeat):
# 
#             plot_y_val = all_vals[:, i_fun, i_repeat] / relative_to_array # Broadcast
#             plot_y_val /= all_y_scalings[i_fun]
#             
#             if all_xvalues is None:
#                 if 'n' in all_args:
#                     plot_x_val = all_sizes * all_x_scalings[i_fun]
#                 else:
#                     raise ValueError # ????
#             else:   
#                 plot_x_val = all_xvalues[:, i_fun, i_repeat] / all_x_scalings[i_fun]
#             
            
            # TODO : Adapt this
#             if transform in ["pol_growth_order", "pol_cvgence_order"]:
#                 
#                 transformed_plot_y_val = np.zeros_like(plot_y_val)
#                 
#                 for i_size in range(1,n_sizes):
#                     
#                     ratio_y = plot_y_val[i_size] / plot_y_val[i_size-1]
#                     ratio_x = plot_x_val[i_size] / plot_x_val[i_size-1]
#                     
#                     try:
#                         transformed_plot_y_val[i_size] = math.log(ratio_y) / math.log(ratio_x)
# 
#                     except:
#                         transformed_plot_y_val[i_size] = np.nan
#                         
#                 transformed_plot_y_val[0] = np.nan
#                                 
#                 plot_y_val = transformed_plot_y_val
# 
#                 if transform == "pol_cvgence_order":
#                     plot_y_val = - transformed_plot_y_val
#                 else:
#                     plot_y_val = transformed_plot_y_val
                
                
            # TODO : Adapt this
#             if clip_vals:
# 
#                 for i_size in range(n_sizes):
#             
#                     if plot_y_val[i_size] < plot_ylim[0]:
# 
#                         if stop_after_first_clip:
#                             for j_size in range(i_size,n_sizes):
#                                 plot_y_val[j_size] = np.nan
#                             break
#                         else:
#                             plot_y_val[i_size] = np.nan
#                             
#                     elif plot_y_val[i_size] > plot_ylim[1]:
# 
#                         if stop_after_first_clip:
#                             for j_size in range(i_size,n_sizes):
#                                 plot_y_val[j_size] = np.nan
#                             break
#                         else:
#                             plot_y_val[i_size] = np.nan
                    
#             # TODO : Is this still needed ?
#             mask = isnotfinite(plot_y_val)
#             masked_plot_y_val = np.ma.array(plot_y_val, mask = mask)
# 
#             if not(np.ma.max(masked_plot_y_val) > 0):
#                 
#             alpha_ = max( alpha / (n_repeat**0.8), 1./255)
# 
#             ax.plot(
#                 plot_x_val              ,
#                 plot_y_val              ,
#                 color = color           ,
#                 linestyle = linestyle   ,
#                 marker = pointstyle     ,
#                 alpha = alpha_          ,
#             )


    for i_subplot_grid_x in range(n_subplot_grid_x):
        for i_subplot_grid_y in range(n_subplot_grid_y):

            cur_ax = ax[i_subplot_grid_y, i_subplot_grid_x]

            if plot_legend:
                
                cur_ax.legend(
                    handles = leg_patch[i_subplot_grid_x][i_subplot_grid_y] ,    
                    bbox_to_anchor = (1.05, 1)                              ,
                    loc = legend_location                                   ,
                    borderaxespad = 0.                                      ,
                )
                
            if logx_plot:
                cur_ax.set_xscale('log')
            if logy_plot:
                cur_ax.set_yscale('log')

            if plot_grid:
                cur_ax.grid(True, which="major", linestyle="-")
                cur_ax.grid(True, which="minor", linestyle="dotted")

            if plot_xlim is not None:
                cur_ax.set_xlim(plot_xlim)
                
            if plot_ylim is not None:
                cur_ax.set_ylim(plot_ylim)

            if title is not None:
                cur_ax.set_title(title)
                
            if xlabel is None:
                if (all_xvalues is None):
                    xlabel = name_points
                else:
                    xlabel = ""
                
            if ylabel is None:
                if mode == "timings":
                    ylabel = "Time (s)"
                else:
                    ylabel = ""

            cur_ax.set_xlabel(xlabel)
            cur_ax.set_ylabel(ylabel)

    if show:
        plt.tight_layout()
        return plt.show()