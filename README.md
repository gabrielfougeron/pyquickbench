<p align="center">
  <a href="https://github.com/gabrielfougeron/pyquickbench"><img alt="pyquickbench" src="https://gabrielfougeron.github.io/pyquickbench/_static/plot_icon.png" width="30%"></a>
</p>

[![Platform](https://anaconda.org/conda-forge/pyquickbench/badges/platforms.svg)](https://pypi.org/project/pyquickbench/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyquickbench.svg?style=flat-square)](https://pypi.org/pypi/pyquickbench/)
[![PyPI version](https://badge.fury.io/py/pyquickbench.svg)](https://pypi.org/project/pyquickbench/)
[![Anaconda version](https://anaconda.org/conda-forge/pyquickbench/badges/version.svg)](https://anaconda.org/conda-forge/pyquickbench)

# Pyquickbench

Pyquickbench is an open source, easy to use benchmarking tool written in pure Python. Checkout the [example gallery](https://gabrielfougeron.github.io/pyquickbench/gallery.html) to get an idea of what pyquickbench is capable.

- **Tutorial:** https://gabrielfougeron.github.io/pyquickbench/tutorial.html
- **Documentation:** https://gabrielfougeron.github.io/pyquickbench/
- **Source code:** https://github.com/gabrielfougeron/pyquickbench
- **Bug reports:** https://github.com/gabrielfougeron/pyquickbench/issues
- **Changelog:** https://github.com/gabrielfougeron/pyquickbench/releases/

## Main features

- Timings / repeatability / output benchmarks
- Error handling
- Benchmark results caching
- Multithreaded / multiprocessed benchmarks
- Benchmark timeout
- Multidimensional benchmarks 
- Transformed data plotting (relative values, convergence order, ...)
- Intelligent plots
- Sensible defaults

## Installation

Pyquickbench is available on the [Python Package Index](https://pypi.org/project/pyquickbench/). To install using pip, simply type:

```
pip install pyquickbench
```

Pyquickbench is available on [conda-forge](https://anaconda.org/conda-forge/pyquickbench). To install using conda, simply type:

```
conda install pyquickbench -c conda-forge
```

To install the current development version of pyquickbench from the [github repository](https://github.com/gabrielfougeron/pyquickbench), you can type:

```
pip install git+ssh://git@github.com/gabrielfougeron/pyquickbench.git 
```

## Tests

To run tests locally on your machine, first checkout this reposity and install dependencies using pip:

```
git clone git@github.com:gabrielfougeron/pyquickbench.git
cd pyquickbench
pip install .[tests]
```

Then, run tests using [pytest](https://docs.pytest.org/en/latest/):

```
pytest
```

## License

This software is published under the [BSD 2-Clause License](https://github.com/gabrielfougeron/pyquickbench/blob/main/LICENSE).

## Other open source alternatives

You might like [perfplot](https://github.com/nschloe/perfplot), one of the inspirations for this work.