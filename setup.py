'''
Creates and compiles C code from Cython files

'''

import os
import sys
import shutil
import setuptools
import numpy
import platform

import Cython.Build
import Cython.Compiler
Cython.Compiler.Options.cimport_from_pyx = True # Mandatory for scipy.LowLevelCallable.from_cython
Cython.Compiler.Options.fast_fail = True
Cython.warn.undeclared = True
src_ext = '.pyx'

cython_extnames = [
    "pyquickbench.cython.rankstats"  
]

for opt_key in ['profile','0','1','2','3','fast']:
    
    cmdline_opt = f"-O{opt_key}"
    
    if cmdline_opt in sys.argv:
        opt_lvl = opt_key
        sys.argv.remove(cmdline_opt)
        break

else:
    opt_lvl = '3'

if platform.system() == "Windows":
    
    ignore_warnings_args = [
        # "-Wno-unused-variable",
        # "-Wno-unused-function",
        # "-Wno-incompatible-pointer-types-discards-qualifiers",
        # "-Wno-unused-command-line-argument"
    ] 

    extra_compile_args = {
        "profile" : ["/Od", "/openmp", *ignore_warnings_args],
        "0" : ["/Od", "/openmp", *ignore_warnings_args],
        "1" : ["/Ox", "/openmp", *ignore_warnings_args],
        "2" : ["/O2", "/openmp", *ignore_warnings_args],
        "3" : ["/O2", "/openmp", *ignore_warnings_args],
        "fast" : ["/O2", "/GL", "/openmp", *ignore_warnings_args],
    }[opt_lvl]

    extra_link_args = {
        "profile" : [*ignore_warnings_args],
        "0" : [*ignore_warnings_args],
        "1" : [*ignore_warnings_args],
        "2" : [*ignore_warnings_args],
        "3" : [*ignore_warnings_args],
        "fast" : ["/GL", *ignore_warnings_args],
    }[opt_lvl]

elif platform.system() == "Darwin": # MacOS

    ignore_warnings_args = [
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-Wno-incompatible-pointer-types-discards-qualifiers",
        "-Wno-unused-command-line-argument"
    ] 
    
    std_args = ["-Xpreprocessor", "-std=c99", "-lm"]
    std_link_args = ["-lm", "-lomp"]

    extra_compile_args = {
        "profile" : ["-Og", *std_args, *ignore_warnings_args],
        "0" : ["-O0", *std_args, *ignore_warnings_args],
        "1" : ["-O1", *std_args, *ignore_warnings_args],
        "2" : ["-O2", *std_args, *ignore_warnings_args],
        "3" : ["-O3", *std_args, *ignore_warnings_args],
        "fast" : ["-Ofast", "-flto", "-march=native", *std_args, *ignore_warnings_args],
    }[opt_lvl]
    
    extra_link_args = {
        "profile" : [*std_link_args, *ignore_warnings_args],
        "0" : [*std_link_args, *ignore_warnings_args],
        "1" : [*std_link_args, *ignore_warnings_args],
        "2" : [*std_link_args, *ignore_warnings_args],
        "3" : [*std_link_args, *ignore_warnings_args],
        "fast" : ["-flto", "-march=native", *std_link_args, *ignore_warnings_args],
    }[opt_lvl]

elif platform.system() == "Linux":
    
    ignore_warnings_args = [
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-Wno-incompatible-pointer-types-discards-qualifiers",
        "-Wno-unused-command-line-argument"
    ] 

    if ("PYODIDE" in os.environ): # Building for Pyodide

        extra_compile_args = {
            "profile" : ["-Og",  *ignore_warnings_args],
            "0" : ["-O0",  *ignore_warnings_args],
            "1" : ["-O1",  *ignore_warnings_args],
            "2" : ["-O2",  *ignore_warnings_args],
            "3" : ["-O3",  *ignore_warnings_args],
            "fast" : ["-O3","-ffast-math","-flto",  *ignore_warnings_args],
        }[opt_lvl]


        extra_link_args = {
            "profile" : [*ignore_warnings_args],
            "0" : [*ignore_warnings_args],
            "1" : [*ignore_warnings_args],
            "2" : [*ignore_warnings_args],
            "3" : [*ignore_warnings_args],
            "fast" : ["-flto", *ignore_warnings_args],
        }[opt_lvl]

    else:
        
        std_args = ["-fopenmp", "-lm"]
        std_link_args = ["-lm", "-fopenmp"]

        extra_compile_args = {
            "profile" : ["-Og", *std_args, *ignore_warnings_args],
            "0" : ["-O0", *std_args, *ignore_warnings_args],
            "1" : ["-O1", *std_args, *ignore_warnings_args],
            "2" : ["-O2", *std_args, *ignore_warnings_args],
            "3" : ["-O3", *std_args, *ignore_warnings_args],
            "fast" : ["-Ofast", "-flto", "-march=native", *std_args, *ignore_warnings_args],
        }[opt_lvl]

        extra_link_args = {
            "profile" : [*std_link_args, *ignore_warnings_args],
            "0" : [*std_link_args, *ignore_warnings_args],
            "1" : [*std_link_args, *ignore_warnings_args],
            "2" : [*std_link_args, *ignore_warnings_args],
            "3" : [*std_link_args, *ignore_warnings_args],
            "fast" : ["-flto", "-march=native", *std_link_args, *ignore_warnings_args],
        }[opt_lvl]

else:

    raise ValueError(f"Unsupported platform: {platform.system()}")

cython_filenames = [ ext_name.replace('.','/') + src_ext for ext_name in cython_extnames]

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if platform.system() == "Windows":
    define_macros.append(("FFTW_NO_Complex", 1))
    define_macros.append(("CYTHON_CCOMPLEX", 0))

if opt_lvl == "fast":
    define_macros.append(("CYTHON_WITHOUT_ASSERTIONS", 1))
    define_macros.append(("CYTHON_CLINE_IN_TRACEBACK", 0))

compiler_directives = {
    'cpow' : True   ,
}

if opt_lvl == "profile" : 

    profile_compiler_directives = {
        'profile': True             ,
        'linetrace': True           ,
        'binding': True             ,
        'embedsignature' : True     ,
        'emit_code_comments' : True , 
    }
    compiler_directives.update(profile_compiler_directives)
    
    profile_define_macros = [
        ('CYTHON_TRACE', '1')       ,
        ('CYTHON_TRACE_NOGIL', '1') ,
    ]
    define_macros.extend(profile_define_macros)

else:
    
    compiler_directives.update({
        'wraparound': False         ,
        'boundscheck': False        ,
        'nonecheck': False          ,
        'initializedcheck': False   ,
        'overflowcheck': False      ,
        'overflowcheck.fold': False ,
        'infer_types': True         ,
        'binding' : False           , 
    })

include_dirs = [
    numpy.get_include(),
]

ext_modules = [
    setuptools.Extension(
        name = name,
        sources =  [source],
        include_dirs = include_dirs,
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros  = define_macros ,
    )
    for (name, source) in zip(cython_extnames, cython_filenames, strict = True)
]

if platform.system() == "Windows":
    nthreads = 0
else:
    import multiprocessing
    nthreads = multiprocessing.cpu_count()

ext_modules = Cython.Build.cythonize(
    ext_modules,
    language_level = "3",
    annotate = True,
    compiler_directives = compiler_directives,
    nthreads = nthreads,
    force = ("-f" in sys.argv),
)
    
packages = setuptools.find_packages()

package_data = {key : ['*.h','*.pyx','*.pxd'] for key in packages}
exclude_package_data = {key : ['*.c'] for key in packages}

setuptools.setup(
    ext_modules = ext_modules                   ,
    zip_safe = False                            ,
    packages = packages                         ,
    package_data = package_data                 ,
    exclude_package_data = exclude_package_data ,
)
