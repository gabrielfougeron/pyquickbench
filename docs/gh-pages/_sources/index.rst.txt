Welcome to pyquickbench's documentation!
========================================

Pyquickbench is an open source, easy to use benchmarking tool written in pure Python. 

Main features
=============

- Timings / repeatability / output benchmarks
- Error handling
- Benchmark results caching
- Multithreaded / multiprocessed benchmarks
- Benchmark timeout
- Multidimensional benchmarks 
- Transformed data plotting (relative values, convergence order, ...)
- Intelligent plots
- Sensible defaults

Usage
=====

Checkout the :ref:`Gallery <examples-index>` to get an idea of what pyquickbench is capable.

Installation
============

Pyquickbench is available on the `Python Package Index <https://pypi.org/project/pyquickbench/>`_. To install using pip, simply type:

.. code-block:: sh

    pip install pyquickbench

Pyquickbench is available on `conda-forge <https://conda-forge.org/>`_. To install using conda, simply type:

.. code-block:: sh

    conda install pyquickbench -c conda-forge

To install the current development version of pyquickbench from the `github repository <https://github.com/gabrielfougeron/pyquickbench>`_, you can type:

.. code-block:: sh

    pip install git+ssh://git@github.com/gabrielfougeron/pyquickbench.git 

Tests
=====

To run tests locally on your machine, first checkout this reposity and install dependencies using pip:

.. code-block:: sh

    git clone git@github.com:gabrielfougeron/pyquickbench.git
    cd pyquickbench
    pip install .[tests]

Then, run tests using  `pytest <https://docs.pytest.org/en/latest/>`_

.. code-block:: sh

    pytest



.. toctree::
    :hidden:
    :includehidden:
    :maxdepth: 1

    _build/auto_examples/index
    api
    test-report/test-report 
   