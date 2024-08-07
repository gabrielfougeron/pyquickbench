"""
==========
Benchmarks
==========

The two main functions of :mod:`pyquickbench`.

.. autosummary::
   :toctree: _generated/
   :caption: Benchmarks
   :nosignatures:

   run_benchmark
   plot_benchmark
  
.. _api-time-train:
 
==========
Time Train
==========

Provides rudimentary profiling features to be used with :mod:`pyquickbench`.

.. autosummary::
   :toctree: _generated/
   :caption: Time Train
   :nosignatures:

   TimeTrain
   TimeTrain.toc
   TimeTrain.tictoc
   TimeTrain.to_dict
   
=========
Constants
=========

A few named constants and default values in :mod:`pyquickbench`.

Named axes
==========

.. autosummary::
   :toctree: _generated/
   :caption: Named axes
   :nosignatures:

   default_ax_name
   fun_ax_name
   repeat_ax_name
   out_ax_name
   
Data handling
=============

.. autosummary::
   :toctree: _generated/
   :caption: Data handling
   :nosignatures:

   all_reductions
   all_plot_intents
   all_transforms
   
Default curve styling
=====================

.. autosummary::
   :toctree: _generated/
   :caption: Default curve styling
   :nosignatures:
   
   default_color_list
   default_linestyle_list
   default_pointstyle_list
      
      
"""

from ._benchmark  import run_benchmark, plot_benchmark

from ._defaults   import default_ax_name, fun_ax_name, repeat_ax_name, out_ax_name
from ._defaults   import default_color_list, default_linestyle_list, default_pointstyle_list
from ._defaults   import all_reductions, all_plot_intents, all_transforms
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

   * ``"avg"``       : Average 
   * ``"min"``       : Minimum
   * ``"max"``       : Maximum
   * ``"median"``    : Median
   * ``"sum"``       : Sum
   * ``"logavg"``    : Exponential of average of log
   * ``"first_el"``  : First element in the array
   * ``"last_el"``   : Last element in the array
   * ``"random_el"`` : An element in the array picked randomly with uniform probability
    
   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark` : Plot benchmarks    
   * :class:`pyquickbench.TimeTrain` : Rudimentary profiling features

"""

all_plot_intents = all_plot_intents
"""Available plot intents.

   List of all available plot intents to be used in :func:`pyquickbench.plot_benchmark`.

   * ``"single_value"``       : A single value is plotted. Requires either ``single_values_val`` or ``single_values_idx`` to be set.
   * ``"points"``             : Values are plotted as points along a curve.
   * ``"same"``               : All values are plotted the same way.
   * ``"curve_color"``        : Values are plotted as different curve colors.
   * ``"curve_linestyle"``    : Values are plotted as different curve line styles.
   * ``"curve_pointstyle"``   : Values are plotted as different markers along the curve.
   * ``"violin"``             : Values are plotted as a violin dispersion plot.
   * ``"subplot_grid_x"``     : Values are plotted on different plot aligned horizontally.
   * ``"subplot_grid_y"``     : Values are plotted on different plot aligned vertically.
   * ``"reduction_[red]"``    : Values are reduced before plotting. ``[red]`` can be any string in :data:`pyquickbench.all_reductions`.

   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark` : Plot benchmarks
   * :data:`pyquickbench.all_reductions` : List of available data reductions

"""

all_transforms = all_transforms
"""Available data transformations.

   List of all available data transformations to be used in :func:`pyquickbench.plot_benchmark`.

   * ``"pol_growth_order"``  : Plots an estimate of :math:`\\alpha` based on consecutive measured values if the data scales as :math:`\\approx n^\\alpha`.
   * ``"pol_cvgence_order"`` : Plots an estimate of :math:`\\alpha` based on consecutive measured values if the data scales as :math:`\\approx n^{-\\alpha}`.

   See :ref:`sphx_glr__build_auto_examples_tutorial_06-Transforming_values.py` for usage example.

   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark` : Plot benchmarks

"""

default_color_list = default_color_list
""" Default list of curve colors.

   Can be overriden using the ``color_list`` argument of :func:`pyquickbench.plot_benchmark`.
   
   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark` : Plot benchmarks
   * :ref:`matplotlib:sphx_glr_users_explain_colors_colors.py` in the official matplotlib documentation.

"""

default_linestyle_list = default_linestyle_list
""" Default list of curve linestyles.

   Can be overriden using the ``linestyle_list`` argument of :func:`pyquickbench.plot_benchmark`.

   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark`
   * :ref:`matplotlib:sphx_glr_gallery_lines_bars_and_markers_linestyles.py` in the official matplotlib documentation.
"""

default_pointstyle_list = default_pointstyle_list
""" Default list of curve markers.

   Can be overriden using the ``pointstyle_list`` argument of :func:`pyquickbench.plot_benchmark`.

   .. rubric:: See Also

   * :func:`pyquickbench.plot_benchmark`
   * :mod:`matplotlib:matplotlib.markers`
"""