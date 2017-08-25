pescador
========
[![PyPI](https://img.shields.io/pypi/v/pescador.svg)](https://pypi.python.org/pypi/pescador)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pescador/badges/version.svg)](https://anaconda.org/conda-forge/pescador)
[![Build Status](https://travis-ci.org/pescadores/pescador.svg?branch=master)](https://travis-ci.org/pescadores/pescador)
[![Coverage Status](https://coveralls.io/repos/pescadores/pescador/badge.svg)](https://coveralls.io/r/pescadores/pescador)
[![Documentation Status](https://readthedocs.org/projects/pescador/badge/?version=latest)](https://readthedocs.org/projects/pescador/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.400700.svg)](https://doi.org/10.5281/zenodo.400700)

Pescador is a library for streaming (numerical) data, primarily for use in machine learning applications.

Pescador addresses the following use cases:

    - **Hierarchical sampling**
    - **Out-of-core learning**
    - **Parallel streaming**

These use cases arise in the following common scenarios:

    - Say you have three data sources `(A, B, C)` that you want to sample. 
      Pescador can dynamically interleave these sources to provide a randomized stream `D <- (A, B, C)`.
      The distribution over `(A, B, C)` need not be uniform: you can specify any distribution you like!

    - Now, say you have 3000 data sources that you want to sample, and they're too large to all fit in RAM at
      once.
      Pescador makes it easy to interleave these sources while maintaining a small `working set`.
      Not all sources are simultaneously active, but Pescador manages the working set so you don't have to.

    - If loading data incurs substantial latency (e.g., due to storage access or pre-processing), this can slow down processing.
      Pescador makes it easy to do this seamlessly in a background process, so that your main thread can continue working.


Want to learn more? [Read the docs!](http://pescador.readthedocs.org)


Installation
============

Pescador can be installed from PyPI through `pip`:
```
pip install pescador
```
or with `conda` using the `conda-forge` channel:
```
conda install -c conda-forge pescador
```
