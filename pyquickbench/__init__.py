"""
==========
Benchmarks
==========

The two main functions of :mod:`pyquickbench`.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :recursive:

   run_benchmark
   plot_benchmark
   
==========
Time Train
==========

Provides rudimentary profiling features to be used with :mod:`pyquickbench`.

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

A few named constants and default values in :mod:`pyquickbench`.

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
   * "sum"       : Sum
   * "logavg"    : Exponential of average of log
   * "first_el"  : First element in the array
   * "last_el"   : Last element in the array
   * "random_el" : An element in the array picked randomly with uniform probability
    
   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark` : Plot benchmarks    
   * :class:`pyquickbench.TimeTrain` : Rudimentary profiling features

"""

all_plot_intents = all_plot_intents
"""Available plot intents.

   List of all available plot intents to be used in :func:`pyquickbench.plot_benchmark`.

   * "single_value"     : A single value is plotted. Requires either ``single_values_val`` or ``single_values_idx`` to be set.
   * "points"           : Values are plotted as points along a curve.
   * "same"             : All values are plotted the same way.
   * "curve_color"      : Values are plotted as different curve colors.
   * "curve_linestyle"  : Values are plotted as different curve line styles.
   * "curve_pointstyle" : Values are plotted as different markers along the curve.
   * "subplot_grid_x"   : Values are plotted on different plot aligned horizontally.
   * "subplot_grid_y"   : Values are plotted on different plot aligned vertically.
   * "reduction_``red``"    : Values are reduced before plotting. ``red`` can be any string in :data:`pyquickbench.all_reductions`.

   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark` : Plot benchmarks
   * :data:`pyquickbench.all_reductions` : List of available data reductions

"""
