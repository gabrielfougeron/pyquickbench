"""
==========
Benchmarks
==========

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :recursive:

   run_benchmark
   plot_benchmark
   
==========
Time Train
==========

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :recursive:

   TimeTrain
   TimeTrain.toc
   TimeTrain.to_dict
   
=========
Constants
=========

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :recursive:
   
   default_ax_name
   fun_ax_name
   repeat_ax_name
   out_ax_name
   all_reductions
   all_plot_intents
   
"""

from ._benchmark  import run_benchmark, plot_benchmark

from ._defaults   import default_ax_name, fun_ax_name, repeat_ax_name, out_ax_name
from ._defaults   import all_reductions, all_plot_intents
from ._time_train import TimeTrain


# Ugly hack to define Sphinx docstrings for variables
default_ax_name = default_ax_name
"""The default value for benchmarked functions arguments."""

fun_ax_name = fun_ax_name
"""The axis name for functions to be benchmarked."""

repeat_ax_name = repeat_ax_name
"""The axis name for reapeated benchmarks."""

out_ax_name = out_ax_name
"""The axis name for function output."""

all_reductions = all_reductions
"""Available data reductions.

   List of all available data reductions to be used in :func:`pyquickbench.plot_benchmark` or :class:`pyquickbench.TimeTrain`.

   * "avg"       : Average 
   * "min"       : Minimum
   * "max"       : Maximum
   * "median"    : Median
   * "sum"       : Sum  x
   * "logavg"    : Exponential of average of log
    
   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark` : Plot benchmarks    
   * :class:`pyquickbench.TimeTrain` : Plot benchmarks    

"""

all_plot_intents = all_plot_intents
"""Available plot intents.

   List of all available plot intents to be used in :func:`pyquickbench.plot_benchmark`.

   * "single_value"     : A single value is drawn 
   * "points"           : Minimum
   * "same"             : Maximum
   * "curve_color"      : Median
   * "curve_linestyle"  : Sum  x
   * "curve_pointstyle" : Exponential of average of log
   * "subplot_grid_x"   : Exponential of average of log
   * "subplot_grid_y"   : Exponential of average of log
   * "reduction_***"    : Exponential of average of log

   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark` : Plot benchmarks
   * :data:`pyquickbench.all_reductions` : List of available data reductions
   * 

"""
