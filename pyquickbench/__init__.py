"""
============
Pyquickbench
============

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :recursive:

   run_benchmark
   plot_benchmark
   
   TimeTrain
   TimeTrain.toc
   TimeTrain.to_dict


"""

from ._benchmark  import run_benchmark, plot_benchmark
from ._defaults   import default_ax_name, fun_ax_name, repeat_ax_name, out_ax_name
from ._time_train import TimeTrain
