import os
import numpy as np
import numpy.typing
import timeit
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure
import typing
import inspect

def isnotfinite(arr):
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res

def _mem_shift(idx, shape):
    
    res = 0
    prod_shapes = 1
    
    # for i in reversed(range(len(shape))): # less efficient : https://stackoverflow.com/questions/7286365/print-a-list-in-reverse-order-with-range/44519681#44519681
    for i in range(len(shape)-1,-1,-1):
        res += i*prod_shapes
        prod_shapes *= shape[i]
        
    return res

default_color_list = list(mpl.colors.TABLEAU_COLORS)
default_color_list.append(mpl.colors.BASE_COLORS['b'])
default_color_list.append(mpl.colors.BASE_COLORS['g'])
default_color_list.append(mpl.colors.BASE_COLORS['r'])
default_color_list.append(mpl.colors.BASE_COLORS['m'])
default_color_list.append(mpl.colors.BASE_COLORS['k'])

default_linestyle_list = [
     ('solid', 'solid')             ,   # Same as (0, ()) or '-'
     ('dotted', 'dotted')           ,   # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed')           ,   # Same as '--'
     ('dashdot', 'dashdot')         ,   # Same as '-.'
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
    
    sig = inspect.signature(fun)
    for param in sig.parameters.values():
        if param.kind == param.POSITIONAL_ONLY:
            raise ValueError(f'Input argument {param} to provided function {fun.__name__} is positional only. Positional-only arguments are unsupported')
    
    setup_vars = setup(*args)

    if isinstance(setup_vars, dict):
        setup_vars_dict = setup_vars
    else:
        setup_vars_dict = {setup_vars[i][0] : setup_vars[i][1] for i in range(len(setup_vars))}
        
    return setup_vars_dict

def _load_benchmark_file(filename, all_sizes_in, shape):
    
    file_base, file_ext = os.path.splitext(filename)
    
    if file_ext == '.npy':
        all_vals = np.load(filename)    

        BenchmarkUpToDate = True
        assert all_vals.ndim == len(shape)
        for loaded_axis_len, expected_axis_len in zip(all_vals.shape, shape.values):
            BenchmarkUpToDate = BenchmarkUpToDate and (loaded_axis_len == expected_axis_len)

        
#     elif file_ext == '.npz':
#         file_content = np.load(filename)
# 
#         all_vals = file_content['all_vals']
# 
#         BenchmarkUpToDate = True
#         assert all_vals.ndim == 3
#         BenchmarkUpToDate = BenchmarkUpToDate and (all_vals.shape[0] == shape[0])
#         BenchmarkUpToDate = BenchmarkUpToDate and (all_vals.shape[1] == shape[1])
#         BenchmarkUpToDate = BenchmarkUpToDate and (all_vals.shape[2] == shape[2])
#         
#         all_sizes = file_content['all_sizes']
#         assert all_sizes.ndim == 1
#         BenchmarkUpToDate = BenchmarkUpToDate and (all_sizes.shape[0] == shape[0])
#         BenchmarkUpToDate = BenchmarkUpToDate and np.all(all_sizes == all_sizes_in)
#     
    else:
        raise ValueError(f'Unknown file extension {file_ext}')

    return all_vals, BenchmarkUpToDate

def _save_benchmark_file(filename, all_vals, all_sizes):
    
    file_bas, file_ext = os.path.splitext(filename)
    
    if file_ext == '.npy':
        np.save(filename, all_vals)    
        
    # elif file_ext == '.npz':
    #     np.savez(
    #         filename                ,
    #         all_vals = all_vals     ,
    #         all_sizes = all_sizes   ,
    #     )
    # 
    else:
        raise ValueError(f'Unknown file extension {file_ext}')
    
def _build_args_shapes(all_args, all_funs, n_repeat):
    
    if not(isinstance(all_args, dict)):
        all_args = {'n': all_args}
    
    assert not('fun' in all_args)
    assert not('repeats' in all_args)
    
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

        all_vals = np.zeros(list(res_shape.values()))

        if mode == "timings":

            for i_args, args in zip(
                itertools.product(*[range(i) for i in args_shape.values()])   ,
                itertools.product(*list(all_args.values()))     ,
            ):

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

                            all_vals[i_args, i_fun, 0] = est_time / n_timeit_0dot2
                            
                        else:
                            # Full benchmark is run

                            n_timeit = math.ceil(n_timeit_0dot2 * time_per_test / est_time / n_repeat)

                            times = Timer.repeat(
                                repeat = n_repeat,
                                number = n_timeit,
                            )
                            
                            all_vals[i_args, i_fun, :] = np.array(times) / n_timeit

                    except Exception as exc:
                        if StopOnExcept:
                            raise exc
                        
                        all_vals[i_args, i_fun, :].fill(np.nan)
                        
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
                            
                        all_vals[i_args, i_fun, i_repeat] = out_val
                        
        else:
            
            raise ValueError(f'Unknown mode {mode}')

        # Sort values along "repeat" axis, taking care of nans.
        idx = np.argsort(all_vals, axis=-1)
        all_vals = np.take_along_axis(all_vals, idx, axis=-1)
        
        if Save_timings_file:
            _save_benchmark_file(filename, all_vals, args_shape)
            
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
    'same'              ,
    'single_value'      ,
    'points'            ,
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
    all_names               : typing.Iterable[str] | None       = None                      ,
    val_plot_intent         : typing.Iterable[str] | None       = None                      ,        
    mode                    : str                               = "timings"                 ,
    all_xvalues             : np.typing.ArrayLike | None        = None                      ,
    # all_x_scalings          : np.typing.ArrayLike | None        = None                  ,
    # all_y_scalings          : np.typing.ArrayLike | None        = None                  ,
    color_list              : list                              = default_color_list        ,
    linestyle_list          : list                              = default_linestyle_list    ,
    pointstyle_list         : list                              = default_pointstyle_list   ,
    logx_plot               : bool | None                       = None                      ,
    logy_plot               : bool | None                       = None                      ,
    plot_ylim               : tuple | None                      = None                      ,
    plot_xlim               : tuple | None                      = None                      ,
    clip_vals               : bool                              = False                     ,
    stop_after_first_clip   : bool                              = False                     ,
    show                    : bool                              = False                     ,
    fig                     : matplotlib.figure.Figure | None   = None                      ,
    ax                      : plt.Axes | None                   = None                      ,
    title                   : str | None                        = None                      ,
    xlabel                  : str | None                        = None                      ,
    ylabel                  : str | None                        = None                      ,
    plot_legend             : bool                              = True                      ,
    plot_grid               : bool                              = True                      ,
    transform               : str | None                        = None                      ,
    alpha                   : float                             = 1.                        ,
    relative_to             : np.typing.ArrayLike | None        = None                      ,
) -> None :

    
    all_args, _, args_shape, res_shape = _build_args_shapes(all_args, all_funs, all_vals.shape[-1])

    assert all_vals.ndim == len(res_shape)
    for loaded_axis_len, expected_axis_len in zip(all_vals.shape, res_shape.values):
        assert loaded_axis_len == expected_axis_len

    n_funs = res_shape['fun']
    n_repeat = res_shape['repeat']

    if all_names is None:
        
        if all_funs is None:
            all_names_list = ['Anonymous function']*n_funs

        else:
            
            all_names_list = []
            
            if isinstance(all_funs, dict):
                for name, fun in all_funs.items():
                    all_names_list.append(name)
            
            else:    
                for fun in all_funs:
                    all_names_list.append(getattr(fun, '__name__', 'Anonymous function'))
                    
    else:
        all_names_list = [name for name in all_names]
    
    assert n_funs == len(all_names_list)
    
    if val_plot_intent is None:
        
        val_plot_intent = {name: 'points' if (i==0) else 'curve_color' for i, name in enumerate(all_args)}
        val_plot_intent['fun'] = 'curve_color'
        val_plot_intent['repeat'] = 'same'

    else:
        
        assert isinstance(val_plot_intent, dict)
        assert len(val_plot_intent) == all_vals.ndim
        for name, intent in val_plot_intent.items():
            assert name in all_args
            assert intent in all_plot_intents
            
        assert 'fun' in val_plot_intent
        assert 'repeat' in val_plot_intent
    
    n_points = 0
    idx_points = -1
    
    'same'              ,
    'single_value'      ,
    'points'            ,
    'curve_color'       ,
    'curve_linestyle'   ,
    'curve_pointstyle'  ,
    'subplot_grid_x'    ,
    'subplot_grid_y'    ,
    'reduction_avg'     ,
    
    idx_same = []
    idx_single_value = []
    idx_curve_color = []
    idx_curve_linestyle = []
    idx_curve_pointstyle = []
    idx_subplot_grid_x = []
    idx_subplot_grid_y = []
    idx_reduction_avg = []
    
    for i, name in enumerate(val_plot_intent):
        
        if name == 'points':
            n_points += 1
            idx_points = i
            
    if (n_points != 1):
        raise ValueError("There should be exactly one 'points' value plot intent")
    
    
    
    
    
    
    
    
    
    
    
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

        dpi = 150
        figsize = (1600/dpi, 800/dpi)

        fig = plt.figure(
            figsize = figsize,
            dpi = dpi   ,
        )  
        ax = fig.add_subplot(1,1,1)
        
    
    # TODO : Adapt this
    if (relative_to is None):
        relative_to_array = np.ones(list(args_shape.values()))
        
    else:
        raise NotImplementedError
        if isinstance(relative_to, np.ndarray):
            relative_to_array = relative_to
            assert relative_to_array.shape == tuple(args_shape.values())
        
        else:
            
            if isinstance(relative_to, int):
                relative_to_idx = relative_to
                
            elif isinstance(relative_to, str):
                try:
                    relative_to_idx = all_names_list.index(relative_to)
                except ValueError:
                    raise ValueError(f'{relative_to} is not a known name')
            else:
                raise ValueError(f'Invalid relative_to argument {relative_to}')
            
            relative_to_array = (np.sum(all_vals[:, relative_to_idx, :], axis=1) / n_repeat)
        
        
    n_colors = len(color_list)
    n_linestyle = len(linestyle_list)
    n_pointstyle = len(pointstyle_list)

    if clip_vals and (plot_ylim is None):
        raise ValueError('Need a range to clip values')

    leg_patch = []
    for i_fun in range(n_funs):

        color = color_list[i_color % n_colors]
        linestyle = linestyle_list[i_linestyle % n_linestyle]
        pointstyle = pointstyle_list[i_pointstyle % n_pointstyle]
        
        leg_patch.append(
            mpl.patches.Patch(
                color = color                   ,
                label = all_names_list[i_fun]   ,
                linestyle = linestyle           ,
            )
        )
        
        
        
        
        
        
        

        for i_repeat in range(n_repeat):

            plot_y_val = all_vals[:, i_fun, i_repeat] / relative_to_array # Broadcast
            plot_y_val /= all_y_scalings[i_fun]
            
            if all_xvalues is None:
                if 'n' in all_args:
                    plot_x_val = all_sizes * all_x_scalings[i_fun]
                else:
                    raise ValueError # ????
            else:   
                plot_x_val = all_xvalues[:, i_fun, i_repeat] / all_x_scalings[i_fun]
            
            
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
            alpha_ = max( alpha / (n_repeat**0.8), 1./255)

            ax.plot(
                plot_x_val              ,
                plot_y_val              ,
                color = color           ,
                linestyle = linestyle   ,
                marker = pointstyle     ,
                alpha = alpha_          ,
            )

    if plot_legend:
        ax.legend(
            handles=leg_patch,    
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
        )

    if logx_plot is None:
        logx_plot = (transform is None)
        
    if logy_plot is None:
        logy_plot = (transform is None)
        
    if logx_plot:
        ax.set_xscale('log')
    if logy_plot:
        ax.set_yscale('log')

    if plot_grid:
        ax.grid(True, which="major", linestyle="-")
        ax.grid(True, which="minor", linestyle="dotted")

    if plot_xlim is not None:
        ax.set_xlim(plot_xlim)
        
    if plot_ylim is not None:
        ax.set_ylim(plot_ylim)

    if title is not None:
        ax.set_title(title)
        
    if xlabel is None:
        if (all_xvalues is None):
            xlabel = "n"
        else:
            xlabel = ""
        
    if ylabel is None:
        if mode == "timings":
            ylabel = "Time (s)"
        else:
            ylabel = ""

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        plt.tight_layout()
        return plt.show()