"""
Time benchmark of built-in hashing in Python
============================================
"""

# %% 
#
# This is a benchmark of different ways to perform inplace conjugation of a complex numpy array.

# sphinx_gallery_start_ignore

import os
import sys
import multiprocessing
import itertools

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

import pyquickbench
import numpy as np
import random
def randbytes(n):
    return {'data':random.randbytes(n)}

import hashlib

def md5(data):
    return hashlib.md5(data)

def sha1(data):
    return hashlib.sha1(data)

def sha224(data):
    return hashlib.sha224(data)

def sha256(data):
    return hashlib.sha256(data)

def sha384(data):
    return hashlib.sha384(data)

def sha512(data):
    return hashlib.sha512(data)

def sha3_224(data):
    return hashlib.sha3_224(data)

def sha3_256(data):
    return hashlib.sha3_256(data)

def sha3_384(data):
    return hashlib.sha3_384(data)

def sha3_512(data):
    return hashlib.sha3_512(data)

all_funs = [
    md5         ,
    sha1        ,
    sha224      ,
    sha256      ,
    sha384      ,
    sha512      ,
    sha3_224    ,
    sha3_256    ,
    sha3_384    ,
    sha3_512    ,
]

all_sizes = np.array([2**n for n in range(25)])
basename = 'Hashing_bench'
timings_filename = os.path.join(timings_folder, basename+'.npz')

n_repeat = 1

all_values = pyquickbench.run_benchmark(
    all_sizes                   ,
    all_funs                    ,
    setup = randbytes           ,
    n_repeat = n_repeat         ,
    filename = timings_filename ,
    ShowProgress=True           ,
)

pyquickbench.plot_benchmark(
    all_values                              ,
    all_sizes                               ,
    all_funs                                ,
    show = True                             ,
    title = 'Built-in hashing in Python'    ,
)

# %% 
# 

relative_to_val = {pyquickbench.fun_ax_name:"sha1"}

pyquickbench.plot_benchmark(
    all_values                              ,
    all_sizes                               ,
    all_funs                                ,
    relative_to_val = relative_to_val       ,
    show = True                             ,
    title = 'Built-in hashing in Python'    ,
    ylabel = 'Time relative to sha1'        ,
)


