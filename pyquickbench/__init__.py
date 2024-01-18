"""
============
Pyquickbench
============

Pyquickbench's public API is only made up of two functions.

.. autosummary::
   :toctree: generated/

   run_benchmark
   plot_benchmark

"""

from ._benchmark import run_benchmark, plot_benchmark
from ._defaults import default_ax_name, fun_ax_name, repeat_ax_name
