import os
import math
import itertools
import typing
import warnings
    
import numpy as np
import numpy.typing
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure
import tqdm
import tqdm.notebook

from pyquickbench._utils import (
    _prod_rel_shapes        ,
    _check_sig_no_pos_only  ,
    _get_rel_idx_from_maze  ,
    _in_ipynb               ,
    _load_benchmark_file    ,
    _save_benchmark_file    ,
    _build_args_shapes      ,
    _build_out_names        ,
    FakeProgressBar         ,
    AllPoolExecutors        ,
    _measure_output         ,
    _measure_timings        ,
    _values_reduction       ,
    _build_product_legend   ,
    _choose_idx_val         ,
    _treat_future_result    ,
    _arg_names_list_to_idx  ,
    _product                ,
    _count_Truthy           ,
)

from pyquickbench._defaults import *

def run_benchmark(
    all_args        : typing.Union[dict, typing.Iterable]                   ,
    all_funs        : typing.Union[dict, typing.Iterable]                   ,
    *                                                                       ,
    mode            : str                       = "timings"                 ,
    setup           : typing.Callable[[int], typing.Dict[str, typing.Any]]
                                                = default_setup             ,
    n_repeat        : int                       = 1                         ,
    nproc           : int                       = None                      ,
    pooltype        : typing.Union[str, None]   = None                      ,
    time_per_test   : float                     = 0.2                       ,
    filename        : typing.Union[str, None]   = None                      ,
    ForceBenchmark  : bool                      = False                     ,
    PreventBenchmark: bool                      = False                     ,
    StopOnExcept    : bool                      = False                     ,
    ShowProgress    : bool                      = False                     ,
    WarmUp          : bool                      = False                     ,
    MonotonicAxes   : list                      = []                        ,
    timeout         : float                     = 1.                        ,
    show            : bool                      = False                     ,
    **plot_kwargs   : typing.Dict[str, typing.Any]                          ,
) -> typing.Union[np.typing.NDArray[np.float64], None] :
    """ Runs a full benchmark.

    Parameters
    ----------
    all_args : dict | typing.Iterable
        Describes the arguments to be given to the functions in the benchmark.
    all_funs : dict | typing.Iterable
        Functions to be benchmarked.
    mode : str, optional
        Benchmark mode, i.e. target of the benchmark.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_05-Plotting_scalars.py` for usage example.\n
        Possible values: ``"timings"`` or ``"scalar_output"``. By default ``"timings"``.
    setup : callable, optional
        Function that prepares the inputs for the functions to be benchmarked.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_03-Preparing_inputs.py` for usage example.\n
        By default ``lambda n: {pyquickbench.default_ax_name: n}``.
    n_repeat : int, optional
        Number of times to repeat the benchmark for variability studies.\n
        By default ``1``.
    nproc : int, optional
        Number of workers in PoolExecutor.\n
        By default :func:`python:multiprocessing.cpu_count()`.
    pooltype : str, optional
        Type of PoolExecutor.\n
        Possible values: ``"phony"``, ``"thread"`` or ``"process"``.\n  
        By default ``"phony"``.
    time_per_test : float, optional
        Minimum time in seconds for benchmark in ``"timings"`` mode.\n
        By default ``0.2``.
    filename : str | None, optional
        Filename for results caching.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_02-Caching_benchmarks.py` for usage example.\n
        Possible file extensions : ``*.npy`` or ``*.npz``.\n
        By default ``None``.
    ForceBenchmark : bool, optional
        Whether to disregard existing cache and force a full re-run, by default ``False``.
    PreventBenchmark : bool, optional
        Whether to prevent a possibly lengthy full re-run, by default ``False``.
    StopOnExcept : bool, optional
        Whether to interrupt the benchmark if exceptions are thrown, by default ``False``.
    ShowProgress : bool, optional
        Whether to show a progress bar in the CLI during benchmark, by default ``False``.
    WarmUp : bool, optional
        Whether to run the function once without measurement. Can help with jit compilation caching for instance.\n
        By default ``False``.
    MonotonicAxes : list, optional
        List of argument names for which timings are expected to get longer and longer.\n
        By default ``[]``.
    timeout : float, optional
        Time in seconds after which a timing is considered too long.\n
        When a timing reaches this value, all longer timings (as detected by ``MonotonicAxes``) are cancelled.\n
        By default ``1.``.
    show : bool, optional
        Whether to issue a call to :func:`pyquickbench.plot_benchmark` after the benchmark is run, by default ``False``.
    **plot_kwargs :
        Arguments to pass on to :func:`pyquickbench.plot_benchmark` after the benchmark is run.

    Returns
    -------
    np.typing.NDArray[np.float64] | None
        Benchmark results.

    """
    if filename is None:
        Load_timings_file = False
        Save_timings_file = False
        
    else:
        Load_timings_file =  os.path.isfile(filename) and not(ForceBenchmark)
        Save_timings_file = True

    if not(isinstance(all_args, dict)):
        all_args = {default_ax_name: all_args}
        
    if isinstance(all_funs, dict):
        all_funs_list = [fun for fun in all_funs.values()]
    else:    
        all_funs_list = [fun for fun in all_funs]

    if mode in ["timings", "scalar_output"]:
        n_out = 1
    elif mode == "vector_output":
        n_out, all_out_names = _build_out_names(all_args, setup, all_funs_list)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    args_shape, res_shape = _build_args_shapes(all_args, all_funs, n_repeat, n_out)

    MonotonicAxes_idx = _arg_names_list_to_idx(MonotonicAxes, all_args)

    if Load_timings_file:
        
        try:
            
            all_vals, BenchmarkUpToDate = _load_benchmark_file(filename, all_args, res_shape)

            DoBenchmark = not(BenchmarkUpToDate) and not(PreventBenchmark)
            if not(BenchmarkUpToDate) and PreventBenchmark:
                warnings.warn("Found and returned non-matching benchmark file because PreventBenchmark is True")

        except Exception as exc:
            
            if StopOnExcept:
                raise exc
            
            all_vals = None
            DoBenchmark = not(PreventBenchmark)
            
    else:
        
        all_vals = None
        DoBenchmark = not(PreventBenchmark)

    if DoBenchmark:

        for fun in all_funs_list:
            _check_sig_no_pos_only(fun)
        
        all_vals = np.full(list(res_shape.values()), -1.) # Negative values
        
        total_iterations = _product(args_shape.values())
        
        if ShowProgress:
            if (_in_ipynb()):
                progress_bar = tqdm.notebook.tqdm
            else:
                progress_bar = tqdm.tqdm
        else:
            progress_bar = FakeProgressBar

        if pooltype ==  None:
            if nproc is None:
                pooltype = "phony"
            else:
                pooltype = "process"                
        else:
            if nproc is None:
                if MULTIPROCESSING_AVAILABLE:
                    nproc = multiprocessing.cpu_count()
                else:
                    nproc = 1
                    
        try:
            PoolExecutor = AllPoolExecutors[pooltype]
        except KeyError:
            raise ValueError(f"Unknown pooltype {pooltype}. Available pootypes: {list(AllPoolExecutors.keys())}")
                    
        if mode == "timings":
            
            if (pooltype != "phony"):
                warnings.warn("Concurrent execution is unwise in timings mode as it will mess up the timings.")
                
            measure_fun = _measure_timings
            extra_submit_args = (setup, all_funs_list, n_repeat, time_per_test, StopOnExcept, WarmUp, all_vals)
            
        elif mode in ["scalar_output", "vector_output"]: 
        
            measure_fun = _measure_output
            extra_submit_args = (setup, all_funs_list, n_repeat, n_out, StopOnExcept)
        
        else:
                
            raise ValueError(f'Unknown mode {mode}')
            
        with progress_bar(total = total_iterations) as progress, PoolExecutor(nproc) as executor:

            for i_args, args in zip(
                itertools.product(*[range(i) for i in args_shape.values()]) ,
                itertools.product(*list(all_args.values()))                 ,
            ):
                
                future = executor.submit(
                    measure_fun,
                    i_args, args, *extra_submit_args
                )
                
                setattr(future, "i_args", i_args)
                setattr(future, "StopOnExcept", StopOnExcept)
                setattr(future, "all_vals", all_vals)
                setattr(future, "timeout", timeout)
                setattr(future, "MonotonicAxes_idx", MonotonicAxes_idx)

                future.add_done_callback(_treat_future_result)
                future.add_done_callback(lambda _: progress.update(1))

        # Sort values along repeat_ax_name axis, taking care of nans.
        idx = np.argsort(all_vals, axis=-2)
        all_vals = np.take_along_axis(all_vals, idx, axis=-2)
        
        if Save_timings_file:
            _save_benchmark_file(filename, all_vals, all_args)
            
    # print(f'{all_vals = }')
    
    if all_vals is None:
        warnings.warn("Benchmark was neither loaded not run")
            
    if show:
        return plot_benchmark(
            all_vals            ,
            all_args            ,
            all_funs            ,
            mode = mode         ,
            setup = setup       ,
            show = show         ,
            **plot_kwargs       ,
        )

    return all_vals

def plot_benchmark(
    all_vals                : np.typing.ArrayLike   ,
    all_args                : typing.Union[dict, typing.Iterable]                           ,
    all_funs                : typing.Union[
                                typing.Dict[str, callable]      ,
                                typing.Iterable[str]            ,
                                None                            ,
                            ]                                   = None                      ,
    *                                                                                       ,
    all_fun_names           : typing.Union[
                                typing.Iterable[str]            ,
                                None                            ,
                            ]                                   = None                      ,
    all_out_names           : typing.Union[
                                typing.Iterable[str]            ,
                                None                            ,
                            ]                                   = None                      ,
    plot_intent             : typing.Union[
                                typing.Iterable[str]            ,
                                None                            ,
                            ]                                   = None                      ,        
    mode                    : str                               = "timings"                 ,
    setup                   : typing.Callable[[int], typing.Dict[str, typing.Any]]
                                                                = None                      ,
    all_xvalues             : typing.Union[
                                np.typing.ArrayLike             ,
                                None                            ,
                            ]                                   = None                      ,
    color_list              : list                              = default_color_list        ,
    linestyle_list          : list                              = default_linestyle_list    ,
    pointstyle_list         : list                              = default_pointstyle_list   ,
    alpha                   : float                             = 1.                        ,
    single_values_idx       : typing.Union[dict, None]          = None                      ,         
    single_values_val       : typing.Union[dict, None]          = None                      ,         
    logx_plot               : typing.Union[bool, None]          = None                      ,
    logy_plot               : typing.Union[bool, None]          = None                      ,
    plot_xlim               : typing.Union[tuple, None]         = None                      ,
    plot_ylim               : typing.Union[tuple, None]         = None                      ,
    show                    : bool                              = False                     ,
    return_empty_plot       : bool                              = False                     ,
    fig                     : typing.Union[
                                matplotlib.figure.Figure        ,
                                None                            ,
                            ]                                   = None                      ,
    ax                      : typing.Union[
                                np.ndarray[plt.Axes]            ,
                                plt.Axes                        ,
                                None                            ,
                            ]                                   = None                      ,
    dpi                     : int                               = 150                       ,
    pxl_per_plot_x          : int                               = 1600                      ,
    pxl_per_plot_y          : int                               = 800                       ,
    sharex                  : bool                              = True                      ,
    sharey                  : bool                              = False                     ,
    title                   : typing.Union[str, None]           = None                      ,
    xlabel                  : typing.Union[str, None]           = None                      ,
    ylabel                  : typing.Union[str, None]           = None                      ,
    plot_legend             : typing.Union[None, bool, dict]    = None                      ,
    ProductLegend           : bool                              = False                     ,
    legend_location         : str                               = 'upper left'              ,
    plot_grid               : bool                              = True                      ,
    relative_to_idx         : typing.Union[dict, None]          = None                      ,         
    relative_to_val         : typing.Union[dict, None]          = None                      , 
    transform               : typing.Union[str, None]           = None                      ,
    clip_vals               : bool                              = False                     ,
    stop_after_first_clip   : bool                              = False                     ,
    filename                : typing.Union[str, None]           = None                      ,
) -> typing.Union[tuple[matplotlib.figure.Figure, np.ndarray[plt.Axes]], None] :    
    """Plots benchmarks results

    Parameters
    ----------
    all_vals : np.typing.ArrayLike
        Benchmark results as returned by :func:`pyquickbench.run_benchmark`.
    all_args : dict | typing.Iterable
        Describes the arguments given to the functions in the benchmark.
    all_funs : typing.Dict[str, callable] | typing.Iterable[str] | None, optional
        Benchmarked functions, by default ``None``.\n
        Only the ``__name__`` attribute is used here.
    all_fun_names : typing.Iterable[str] | None, optional
        Names of the benchmarked functions, by default ``None``.\n
        In case the functions ``__name__`` attribute is missing or uninformative.
    all_out_names : typing.Iterable[str] | None, optional
        Names of the outputs of the functions if ``mode="vector_output"``, by default ``None``.
    plot_intent : typing.Iterable[str] | None, optional
        Describes how to handle the axes of the benchmark results array ``all_vals``.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_07-Multidimensional_benchmarks.py` for usage examples.\n
        By default ``None``.
    mode : str, optional
        Benchmark mode, i.e. target of the benchmark.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_05-Plotting_scalars.py` for usage example.\n
        Possible values: ``"timings"`` or ``"scalar_output"``. By default ``"timings"``.
    setup : callable, optional
        Function that prepares the inputs for the functions to be benchmarked.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_03-Preparing_inputs.py` for usage example.\n
        Only used if ``all_out_names`` was not provided and ``mode="vector_output"``\n
        By default ``lambda n: {pyquickbench.default_ax_name: n}``.
    all_xvalues : np.typing.ArrayLike | None, optional
        Values to be plotted on the x-axis if those differ from argument values.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_05-Plotting_scalars.py` for usage example.\n 
        By default ``None``.
    color_list : list, optional
        List of colors for plotted curves, by default ``default_color_list``.
    linestyle_list : list, optional
        List of linestyles for plotted curves, by default ``default_linestyle_list``.
    pointstyle_list : list, optional
        List of point markers for plotted curves, by default ``default_pointstyle_list``.
    alpha : float, optional
        Alpha value for transparency of curves in plot.\n
        By default, ``1.`` meaning curves are fully opaque.
    single_values_idx : dict | None, optional
        Indices of benchmarked values to be fixed by a ``plot_intent`` of ``"single_value"``.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_07-Multidimensional_benchmarks.py` for usage example.\n
        By default ``None``.
    single_values_val : dict | None, optional
        Values of benchmark to be fixed by a ``plot_intent`` of ``"single_value"``.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_07-Multidimensional_benchmarks.py` for usage example.\n
        By default ``None``.
    logx_plot : bool | None, optional
        How to override log scaling on the x-axis of the plots, by default ``None``.
    logy_plot : bool | None, optional
        How to override log scaling on the y-axis of the plots, by default ``None``.
    plot_xlim : tuple | None, optional
       How to override limits on the x-axis of the plots, by default ``None``.
    plot_ylim : tuple | None, optional
        How to override limits on the y-axis of the plots, by default ``None``.
    show : bool, optional
        Whether to issue a ``plt.show()`` instead of returning a ``(fig, ax)`` tuple, by default ``False``.
    return_empty_plot : bool, optional
        Whether to prematurely return an empty ``(fig, ax)`` tuple.
        Useful to create an empty plot and reuse it, see :ref:`sphx_glr__build_auto_examples_tutorial_07-Multidimensional_benchmarks.py` for instance.
        By defaut ``False``.
    fig : matplotlib.figure.Figure | None, optional
        User provided :class:`matplotlib:matplotlib.figure.Figure` object.\n 
        Typically, this argument is the result of a former call to :func:`pyquickbench.plot_benchmark`, potentially using argument ``return_empty_plot = True``.\n
        By default ``None``.
    ax : plt.Axes | None, optional
        User provided array of :class:`matplotlib:matplotlib.axes.Axes` objects as returned by :func:`matplotlib:matplotlib.pyplot.subplots`.\n
        Typically, this argument is the result of a former call to :func:`pyquickbench.plot_benchmark`, potentially using argument ``return_empty_plot = True``.\n
        By default ``None``.
    dpi : int, optional
        Output image resolution, by default ``150``.
    pxl_per_plot_x : int, optional
        Output plot width, by default ``1600``.
    pxl_per_plot_y : int, optional
        Output plot height, by default ``800``.
    sharex : bool, optional
        Whether to share plot x-axis, by default ``True``.
    sharey : bool, optional
        Whether to share plot y-axis, by default ``False``.
    title : str | None, optional
        Image title, by default ``None``.
    xlabel : str | None, optional
        Override argument value as a default for plot x label, by default ``None``.
    ylabel : str | None, optional
        Override default for plot y label, by default ``None``.
    plot_legend : bool | dict | None, optional
        Whether to record each axis of the benchmark in a legend, by default ``None``.
    legend_location : str, optional
        Location of plot legend as given to :meth:`matplotlib:matplotlib.axes.Axes.legend`, by default ``'upper left'``.
    plot_grid : bool, optional
        Whether to plot a background grid to each plot, by default ``True``.
    ProductLegend : bool, optional
        Whether to detail every curve in the legend, or aggregate benchmark axes, leading to more concise legends.\n
        By default ``False``.
    relative_to_idx : dict | None, optional
        Indices of benchmarked values against which curves will be plotted.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_06-Transforming_values.py` for usage example.\n
        By default ``None``.
    relative_to_val : dict | None, optional
        Values of benchmark against which curves will be plotted.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_06-Transforming_values.py` for usage example.\n
        By default ``None``.
    transform : str | None, optional
        Data transformation before plotting, by default ``None``.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_06-Transforming_values.py` for usage example.
    clip_vals : bool, optional
        Whether to clip values that are out of bounds. Requires the argument ``plot_ylim`` tu be set explicitely.
        By default ``None``.\n
        See :ref:`sphx_glr__build_auto_examples_tutorial_06-Transforming_values.py` for usage example.
    stop_after_first_clip : bool, optional
        Whether to stop plotting after the first clipped value if ``clip_vals == True``, by default False
    filename : str | None, optional
        If not ``None``, saves resulting figure in ``filename``.\n
        By default ``None``.
    """

    # print(f'{all_vals = }')
    
    all_vals = np.ma.array(all_vals, mask=np.isnan(all_vals))
    
    if not(isinstance(all_args, dict)):
        all_args = {default_ax_name: all_args}
    
    if all_fun_names is None:
        
        if all_funs is None:
            raise ValueError('At least one of all_funs or all_fun_names must be provided')

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
    
    args_shape, res_shape = _build_args_shapes(all_args, all_fun_names_list, all_vals.shape[-2], all_vals.shape[-1])

    if not(isinstance(all_vals, np.ndarray)):
        raise ValueError(f'all_vals should be a np.ndarry. Provided all_vals is a {type(all_vals)} instead.')
    
    if not(all_vals.ndim == len(res_shape)):
        raise ValueError(f'all_vals has the wrong number of dimensions. Received {all_vals.ndim}, expected {len(res_shape)}.')

    for loaded_axis_len, (axis_name, expected_axis_len) in zip(all_vals.shape, res_shape.items()):
        if not(loaded_axis_len == expected_axis_len):
            raise ValueError(f'Axis {axis_name} of the benchmark results has a length of {loaded_axis_len} instead of the expected {expected_axis_len}.')
            
    n_funs = res_shape[fun_ax_name]
    n_repeat = res_shape[repeat_ax_name]
    n_out = res_shape[out_ax_name]
    
    if (all_out_names is None):
        
        if (mode != 'vector_output'):
            
            all_out_names_list = [str(idx) for idx in range(n_out)]

        elif all_funs is None:

            all_out_names_list = [str(idx) for idx in range(n_out)]

        else:
            
            if setup is None:
                all_out_names_list = [str(idx) for idx in range(n_out)]
                    
            else:
                
                if isinstance(all_funs, dict):
                    all_funs_list = [fun for fun in all_funs.values()]
                else:    
                    all_funs_list = [fun for fun in all_funs]
                
                _, all_out_names_list = _build_out_names(all_args, setup, all_funs_list)

    else:
        all_out_names_list = [name for name in all_out_names]
    
    assert n_funs == len(all_fun_names_list)

    if not(isinstance(color_list, list)):
        color_list = [color_list]    
    if not(isinstance(linestyle_list, list)):
        linestyle_list = [linestyle_list]    
    if not(isinstance(pointstyle_list, list)):
        pointstyle_list = [pointstyle_list]
    
    n_colors = len(color_list)
    n_linestyle = len(linestyle_list)
    n_pointstyle = len(pointstyle_list)
    
    if not(transform is None):
        if not(transform in all_transforms):
            raise ValueError((f'Unknown transform {transform}. Possible values are: {all_transforms}'))

    if plot_intent is None:
        
        plot_intent = {name: 'points' if (i==0) else 'curve_color' for i, name in enumerate(all_args)}
        plot_intent[fun_ax_name] = 'curve_color'
        plot_intent[repeat_ax_name] = default_reduction
        plot_intent[out_ax_name] = 'curve_linestyle'

    else:
        
        assert isinstance(plot_intent, dict)
        
        if fun_ax_name not in plot_intent:
            plot_intent[fun_ax_name] = 'curve_color'
        if repeat_ax_name not in plot_intent:
            plot_intent[repeat_ax_name] = default_reduction
        if out_ax_name not in plot_intent:
            plot_intent[out_ax_name] = 'curve_linestyle'
        
        if not(len(plot_intent) == all_vals.ndim):  
            raise ValueError(f"Foud {len(plot_intent)} plot_intents. Expecting {all_vals.ndim}")
        
        for name, intent in plot_intent.items():
            if not(name in res_shape):
                raise ValueError(f'Unknown argument {name} in plot_intent')
            
            if not(intent in all_plot_intents):
                raise ValueError(f'Unknown intent {intent} in plot_intent. Possible values are: {all_plot_intents}')

        plot_intent = {name:plot_intent[name] for name in res_shape} 

    if plot_legend is None:
        
        plot_legend = {name: True for name in all_args}
        plot_legend[fun_ax_name] = (n_funs > 1)
        plot_legend[repeat_ax_name] = (n_repeat > 1)
        plot_legend[out_ax_name] = (n_out > 1)

    elif isinstance(plot_legend, bool):
        plot_legend_in = plot_legend

        plot_legend = {name: plot_legend_in for name in all_args}
        plot_legend[fun_ax_name] = plot_legend_in
        plot_legend[repeat_ax_name] = plot_legend_in
        plot_legend[out_ax_name] = plot_legend_in
        
    elif isinstance(plot_legend, dict):
        
        for name in all_args:
            if name not in plot_legend:
                plot_legend[name] = True
        
        if fun_ax_name not in plot_legend:
            plot_legend[fun_ax_name] = (n_funs > 1)        
        if repeat_ax_name not in plot_legend:
            plot_legend[repeat_ax_name] = (n_repeat > 1)        
        if out_ax_name not in plot_legend:
            plot_legend[out_ax_name] = (n_out > 1)
                
            
        if not(len(plot_legend) == all_vals.ndim):  
            raise ValueError(f"Foud {len(plot_legend)} plot_legends. Expecting {all_vals.ndim}")
        
        for name, in_legend in plot_legend.items():
            if not(name in res_shape):
                raise ValueError(f'Unknown argument {name} in plot_legend')
            
            if not(isinstance(in_legend, bool)):
                raise ValueError(f'Expected bool values in plot_legend dict. Got: {type(in_legend)}')
            
    else:
        raise TypeError(f'Could not use plot_legend with type {type(plot_legend)}')

    if not((single_values_idx is None) or (single_values_val is None)):
        raise ValueError("Both single_values_idx and single_values_val were set. Only one of them should be")
    if not((relative_to_idx is None) or (relative_to_val is None)):
        raise ValueError("Both relative_to_idx and relative_to_val were set. Only one of them should be")

    n_points = 0
    idx_points = -1

    idx_all_same = []
    idx_all_single_value = []
    idx_all_curve_color = []
    idx_all_curve_linestyle = []
    idx_all_curve_pointstyle = []
    idx_all_subplot_grid_x = []
    idx_all_subplot_grid_y = []
    idx_all_relative = []
    
    n_reductions = 0
    idx_all_reduction = {}
    for name in all_reductions: # Valitity of reduction key was checked before
        idx_all_reduction[name] = []
    
    idx_single_value = []
    name_curve_color = []
    name_curve_linestyle = []
    name_curve_pointstyle = []
    name_subplot_grid_x = []
    name_subplot_grid_y = []
    idx_relative = []
    
    all_legend_curve_color = []
    all_legend_curve_linestyle = []
    all_legend_curve_pointstyle = []
    all_legend_subplot_grid_x = []
    all_legend_subplot_grid_y = []
    
    for i, (name, value) in enumerate(plot_intent.items()):
        
        if value == 'points':
            n_points += 1
            idx_points = i
            name_points = name
        elif value == 'same':
            idx_all_same.append(i)
        elif value == 'single_value':

            fixed_idx = _choose_idx_val(name, single_values_idx, single_values_val, all_args, all_fun_names_list, all_out_names_list)
            
            if fixed_idx is None:
                warnings.warn("Arguments single_values_idx / single_values_val were not properly set. A sensible default was provided, but please beware.")
                fixed_idx = 0
                
            assert isinstance(fixed_idx, int)
            
            idx_all_single_value.append(i)
            idx_single_value.append(fixed_idx)            
            
        elif value == 'curve_color':
            idx_all_curve_color.append(i)
            name_curve_color.append(name)
            all_legend_curve_color.append(plot_legend[name])
        elif value == 'curve_linestyle':
            idx_all_curve_linestyle.append(i)
            name_curve_linestyle.append(name)
            all_legend_curve_linestyle.append(plot_legend[name])
        elif value == 'curve_pointstyle':
            idx_all_curve_pointstyle.append(i)
            name_curve_pointstyle.append(name)
            all_legend_curve_pointstyle.append(plot_legend[name])
        elif value == 'subplot_grid_x':
            idx_all_subplot_grid_x.append(i)
            name_subplot_grid_x.append(name)
            all_legend_subplot_grid_x.append(plot_legend[name])
        elif value == 'subplot_grid_y':
            idx_all_subplot_grid_y.append(i)
            name_subplot_grid_y.append(name)
            all_legend_subplot_grid_y.append(plot_legend[name])
        elif value.startswith("reduction_"):
            n_reductions += 1
            name = value[10:]
            idx_all_reduction[name].append(i)
        else:
            raise ValueError("This error should never be raised")
            
        relative_idx = _choose_idx_val(name, relative_to_idx, relative_to_val, all_args, all_fun_names_list, all_out_names_list)
        if isinstance(relative_idx, int):
            idx_all_relative.append(i)
            idx_relative.append(relative_idx)   
            
    if (n_points != 1):
        raise ValueError(f"There should be exactly one plot_intent named 'points'. There are currently {n_points}.")
    
    if (n_reductions > 1):
        warnings.warn("Several reductions were requested. These reductions will be applied in the order of the axes of the benchmark. Watch out for surprizing results as reductions might not commute in general")
    
    idx_all_curves = []
    idx_all_curves.extend(idx_all_same)
    idx_all_curves.extend(idx_all_curve_color)
    idx_all_curves.extend(idx_all_curve_linestyle)
    idx_all_curves.extend(idx_all_curve_pointstyle)
    idx_all_curves.extend(idx_all_subplot_grid_x)
    idx_all_curves.extend(idx_all_subplot_grid_y)
    
    idx_all_same                = np.array(idx_all_same             )
    idx_all_curve_color         = np.array(idx_all_curve_color      )
    idx_all_curve_linestyle     = np.array(idx_all_curve_linestyle  )
    idx_all_curve_pointstyle    = np.array(idx_all_curve_pointstyle )
    idx_all_subplot_grid_x      = np.array(idx_all_subplot_grid_x   )
    idx_all_subplot_grid_y      = np.array(idx_all_subplot_grid_y   )
    
    for name in all_reductions:
        idx_all_reduction[name] = np.array(idx_all_reduction[name]  )

    n_curves_color = _prod_rel_shapes(idx_all_curve_color, all_vals.shape)
    n_curves_linestyle = _prod_rel_shapes(idx_all_curve_linestyle, all_vals.shape)
    n_curves_pointstyle = _prod_rel_shapes(idx_all_curve_pointstyle, all_vals.shape)
    n_subplot_grid_x = _prod_rel_shapes(idx_all_subplot_grid_x, all_vals.shape)
    n_subplot_grid_y = _prod_rel_shapes(idx_all_subplot_grid_y, all_vals.shape)
    
    n_curves = n_curves_color * n_curves_linestyle * n_curves_pointstyle

    leg_patch = [[[] for _ in range(n_subplot_grid_y)] for __ in range(n_subplot_grid_x)]

    if clip_vals and (plot_ylim is None):
        raise ValueError('Need a range to clip values')

    if logx_plot is None:
        logx_plot = True
        
    if logy_plot is None:
        logy_plot = (transform is None)

    if (ax is None) or (fig is None):

        figsize = (
            n_subplot_grid_x * pxl_per_plot_x / dpi,
            n_subplot_grid_y * pxl_per_plot_y / dpi,
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
            raise TypeError(f'Non compatible type for argument "ax": {type(ax)}.')
        
    if  return_empty_plot:
        return (fig, ax)    
        
    for idx_curve in itertools.product(*[range(all_vals.shape[i]) for i in idx_all_curves]):
        
        idx_vals = [None] * all_vals.ndim
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

        i_curve = i_color + n_curves_color * (i_linestyle + n_curves_linestyle * i_pointstyle)

        if OnlyThisOnce:
            
            if ProductLegend: 
                        
                n_cat_color, i_cat_color = _count_Truthy(all_legend_curve_color)
                n_cat_linestyle, i_cat_linestyle = _count_Truthy(all_legend_curve_linestyle)
                n_cat_pointstyle, i_cat_pointstyle = _count_Truthy(all_legend_curve_pointstyle)
                
                ncats = n_cat_color + n_cat_linestyle + n_cat_pointstyle
                
                KeyValLegend = (ncats > 1)
            
                label = ''
                label = _build_product_legend(idx_curve_color, all_legend_curve_color, name_curve_color, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, label)
                label = _build_product_legend(idx_curve_linestyle, all_legend_curve_linestyle, name_curve_linestyle, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, label)
                label = _build_product_legend(idx_curve_pointstyle, all_legend_curve_pointstyle, name_curve_pointstyle, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, label)
                label = label[:-2] # Removing final comma
                
                leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                    mpl.lines.Line2D([], []     ,
                        color = color           ,
                        label = label           ,
                        linestyle = linestyle   ,
                        marker = pointstyle     ,
                        markersize = Legend_markersize         ,
                    )
                )

        plot_y_val = _values_reduction(all_vals, idx_vals, idx_points, idx_all_reduction)
        
        if all_xvalues is None:
            if name_points == fun_ax_name:
                plot_x_val = all_fun_names
            elif name_points == repeat_ax_name:
                plot_x_val = np.array(range(n_repeat))
            else:
                plot_x_val = all_args[name_points]
        else:  
            plot_x_val =_values_reduction(all_xvalues, idx_vals, idx_points, idx_all_reduction) 
            plot_x_val = plot_x_val.reshape(-1)

        if len(idx_all_relative) > 0:
            
            for i, j in zip(idx_relative, idx_all_relative):
                idx_vals[j] = i

            relative_plot_y_val = _values_reduction(all_vals, idx_vals, idx_points, idx_all_reduction)
            
            plot_y_val /= relative_plot_y_val

        plot_y_val = plot_y_val.reshape(-1)

        npts = len(plot_x_val)
        assert npts == len(plot_y_val)
        
        if transform in ["pol_growth_order", "pol_cvgence_order"]:

            transformed_plot_y_val = np.zeros_like(plot_y_val)
            
            for i_size in range(1,npts):
                
                ratio_y = plot_y_val[i_size] / plot_y_val[i_size-1]
                ratio_x = plot_x_val[i_size] / plot_x_val[i_size-1]
                
                with warnings.catch_warnings(): # Super annoying warning here
                    warnings.simplefilter("ignore")
                    try:
                        transformed_plot_y_val[i_size] = math.log(ratio_y) / math.log(ratio_x)

                    except:
                        transformed_plot_y_val[i_size] = np.nan
                    
            transformed_plot_y_val[0] = np.nan
                            
            plot_y_val = transformed_plot_y_val

            if transform == "pol_cvgence_order":
                plot_y_val = - transformed_plot_y_val
            else:
                plot_y_val = transformed_plot_y_val
            
        if clip_vals:

            for i_size in range(npts):
        
                if plot_y_val[i_size] < plot_ylim[0]:

                    if stop_after_first_clip:
                        for j_size in range(i_size,npts):
                            plot_y_val[j_size] = np.nan
                        break
                    else:
                        plot_y_val[i_size] = np.nan
                        
                elif plot_y_val[i_size] > plot_ylim[1]:

                    if stop_after_first_clip:
                        for j_size in range(i_size,npts):
                            plot_y_val[j_size] = np.nan
                        break
                    else:
                        plot_y_val[i_size] = np.nan
        
        if isinstance(plot_x_val[0], str):
            
            logx_plot = False
            
            tick_width = 0.8
            bar_width = tick_width / n_curves
            x_pos_mid = np.arange(0, npts)
            
            x_pos = x_pos_mid + i_curve * bar_width - tick_width/2 + bar_width/2

            cur_ax.bar(
                x_pos                       ,
                plot_y_val                  ,
                width       = bar_width     ,
                color       = color         ,
                linestyle   = linestyle     ,
                alpha       = alpha         ,
            )

            plt.xticks(x_pos_mid, plot_x_val);
        
        else:    
            
            cur_ax.plot(
                plot_x_val                  ,
                plot_y_val                  ,
                color       = color         ,
                linestyle   = linestyle     ,
                marker      = pointstyle    ,
                alpha       = alpha         ,
            )
        
    if not(ProductLegend): 
        
        n_cat_color, i_cat_color = _count_Truthy(all_legend_curve_color)
        n_cat_linestyle, i_cat_linestyle = _count_Truthy(all_legend_curve_linestyle)
        n_cat_pointstyle, i_cat_pointstyle = _count_Truthy(all_legend_curve_pointstyle)
        
        ncats = n_cat_color + n_cat_linestyle + n_cat_pointstyle
        HeaderLegend = (ncats > 1)
             
        for i_subplot_grid_x in range(n_subplot_grid_x):
            for i_subplot_grid_y in range(n_subplot_grid_y):

                cur_ax = ax[i_subplot_grid_y, i_subplot_grid_x]

                HeaderLegend = (n_cat_color == 1)
                KeyValLegend = not(HeaderLegend)
                
                if n_cat_color > 0:
                    
                    if HeaderLegend:
                        
                        leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                            mpl.patches.Patch(
                                color = "white"                             ,
                                label = f'{name_curve_color[i_cat_color]}:' ,
                                alpha = 0.                                  ,
                            )
                        )
                    
                    for i_color, idx_curve_color in enumerate(itertools.product(*[range(all_vals.shape[i]) for i in idx_all_curve_color])):

                        color = color_list[i_color % n_colors]

                        label = ''
                        label = _build_product_legend(idx_curve_color, all_legend_curve_color, name_curve_color, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, label)
                        label = label[:-2] # Removing final comma

                        leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                            mpl.lines.Line2D([], []                 ,
                                color = color                       ,
                                label = label                       ,
                                linestyle = Legend_bland_linestyle  ,
                                marker = Legend_bland_pointstyle    ,
                                markersize = Legend_markersize      ,
                            )
                        )
                        
                    if HeaderLegend:
                        leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                            mpl.patches.Patch(
                                color = "white" ,
                                label = ''      ,
                                alpha = 0.      ,
                            )
                        )
                
                HeaderLegend = (n_cat_linestyle == 1)
                KeyValLegend = not(HeaderLegend)
                
                if n_cat_linestyle > 0:
                    
                    if HeaderLegend:
                        leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                            mpl.patches.Patch(
                                color = "white"                         ,
                                label = f'{name_curve_linestyle[i_cat_linestyle]}:'   ,
                                alpha = 0.                              ,
                            )
                        )
                    
                    for i_linestyle, idx_curve_linestyle in enumerate(itertools.product(*[range(all_vals.shape[i]) for i in idx_all_curve_linestyle])):

                        linestyle = linestyle_list[i_linestyle % n_linestyle]
                        label = ''
                        label = _build_product_legend(idx_curve_linestyle, all_legend_curve_linestyle, name_curve_linestyle, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, label)
                        label = label[:-2] # Removing final comma
                        leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                            mpl.lines.Line2D([], []                 ,
                                color = Legend_bland_color          ,
                                label = label                       ,
                                linestyle = linestyle               ,
                                marker = Legend_bland_pointstyle    ,
                                markersize = Legend_markersize      ,
                            )
                        )
                    
                    if HeaderLegend:
                        leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                            mpl.patches.Patch(
                                color = "white" ,
                                label = ''      ,
                                alpha = 0.      ,
                            )
                        )
                
                HeaderLegend = (n_cat_pointstyle == 1)
                KeyValLegend = not(HeaderLegend)
                        
                if n_cat_pointstyle > 0:
                    
                    if HeaderLegend:
                        leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                            mpl.patches.Patch(
                                color = "white"                         ,
                                label = f'{name_curve_pointstyle[i_cat_pointstyle]}:'  ,
                                alpha = 0.                              ,
                            )
                        )
                    
                    for i_pointstyle, idx_curve_pointstyle in enumerate(itertools.product(*[range(all_vals.shape[i]) for i in idx_all_curve_pointstyle])):

                        pointstyle = pointstyle_list[i_pointstyle % n_pointstyle]
                        label = ''
                        label = _build_product_legend(idx_curve_pointstyle, all_legend_curve_pointstyle, name_curve_pointstyle, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, label)
                        label = label[:-2] # Removing final comma

                        leg_patch[i_subplot_grid_x][i_subplot_grid_y].append(
                            mpl.lines.Line2D([], []                 ,
                                color = Legend_bland_color          ,
                                label = label                       ,
                                linestyle = Legend_bland_linestyle  ,
                                marker = pointstyle                 ,
                                markersize = Legend_markersize      ,
                            )
                        )

    for i_subplot_grid_x, idx_subplot_grid_x in enumerate(itertools.product(*[range(all_vals.shape[i]) for i in idx_all_subplot_grid_x])):
        for i_subplot_grid_y, idx_subplot_grid_y in enumerate(itertools.product(*[range(all_vals.shape[i]) for i in idx_all_subplot_grid_y])):

            cur_ax = ax[i_subplot_grid_y, i_subplot_grid_x]

            if plot_legend:
                
                cur_ax.legend(
                    handles = leg_patch[i_subplot_grid_x][i_subplot_grid_y] ,    
                    bbox_to_anchor = (1.05, 1)                              ,
                    loc = legend_location                                   ,
                    borderaxespad = 0.                                      ,
                )
                            
            n_cat_subplot_grid_x, i_cat_subplot_grid_x = _count_Truthy(all_legend_subplot_grid_x)
            n_cat_subplot_grid_y, i_cat_subplot_grid_y = _count_Truthy(all_legend_subplot_grid_y)
            ncats = n_cat_subplot_grid_x + n_cat_subplot_grid_y
                
            if (n_subplot_grid_x*n_subplot_grid_y > 1) and (ncats > 0):
                KeyValLegend = True
                ax_title = ''
                ax_title = _build_product_legend(idx_subplot_grid_x, all_legend_subplot_grid_x, name_subplot_grid_x, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, ax_title)
                ax_title = _build_product_legend(idx_subplot_grid_y, all_legend_subplot_grid_y, name_subplot_grid_y, all_args, all_fun_names_list, all_out_names_list, KeyValLegend, ax_title)
                ax_title = ax_title[:-2]
                
            elif title is not None:
                ax_title = title
            else:
                ax_title = ''
                    
            cur_ax.set_title(ax_title)
            
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
                
            if xlabel is None:
                if (all_xvalues is None):
                    xlabel = name_points
                else:
                    xlabel = ""
                
            if ylabel is None:
                if transform in ["pol_growth_order", "pol_cvgence_order"]:
                    ylabel = "order"
                elif not((relative_to_val is None) and (relative_to_idx is None)):
                    ylabel = ""
                elif mode == "timings":
                    ylabel = "Time (s)"
                else:
                    ylabel = ""

            cur_ax.set_xlabel(xlabel)
            cur_ax.set_ylabel(ylabel)

    if title is not None:
        if (n_subplot_grid_x*n_subplot_grid_y) > 1:
            fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    
    if filename is not None:
        plt.savefig(filename)
    
    if show:
        return plt.show()
    else:
        return (fig, ax)
