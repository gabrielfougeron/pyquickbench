"""
Description of the benchmark machine
====================================
"""

# %%
# Description of the machine on which benchmarks were run.

import subprocess
import os

Id = subprocess.check_output(['lshw']).decode('utf-8').split(os.linesep)
for line in Id:
    print(line)
    

# %%
# Numpy configuration informations

import numpy as np
np.show_config()

# %%
# Scipy configuration informations

import scipy
scipy.show_config()