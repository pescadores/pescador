pescador
========
[![PyPI](https://img.shields.io/pypi/v/pescador.svg)](https://pypi.python.org/pypi/pescador)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pescador/badges/version.svg)](https://anaconda.org/conda-forge/pescador)
[![Build Status](https://github.com/pescadores/pescador/actions/workflows/ci.yml/badge.svg)](https://github.com/pescadores/pescador/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pescadores/pescador/graph/badge.svg?token=aCgfizw6O5)](https://codecov.io/gh/pescadores/pescador)
[![Documentation Status](https://readthedocs.org/projects/pescador/badge/?version=latest)](https://readthedocs.org/projects/pescador/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.400700.svg)](https://doi.org/10.5281/zenodo.400700)

Pescador is a library for streaming (numerical) data, primarily for use in machine learning applications.

Pescador addresses the following use cases:

- **Hierarchical sampling**
- **Out-of-core learning**
- **Parallel streaming**

These use cases arise in the following common scenarios:

- Say you have three data sources `(A, B, C)` that you want to sample. 
  For example, each data source could contain all the examples of a particular category.

  Pescador can dynamically interleave these sources to provide a randomized stream `D <- (A, B, C)`.
  The distribution over `(A, B, C)` need not be uniform: you can specify any distribution you like!

- Now, say you have 3000 data sources, each of which may contain a large number of samples.  Maybe that's too much data to fit in RAM at once.

  Pescador makes it easy to interleave these sources while maintaining a small `working set`.
  Not all sources are simultaneously active, but Pescador manages the working set so you don't have to.

- If loading data incurs substantial latency (e.g., due to accessing on-disk storage
  or pre-processing), this can be a problem.
  
  Pescador can seamlessly move data generation into a background process, so that your main thread can continue working.


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
