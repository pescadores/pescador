.. pescador documentation master file, created by
   sphinx-quickstart on Fri Apr  3 10:03:34 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _pescador:

########
Pescador
########

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
      This way, you can process the full data set *out of core*, but using a bounded
      amount of memory.

    - If loading data incurs substantial latency (e.g., due to accessing storage access
      or pre-processing), this can be a problem.

      Pescador can seamlessly move data generation into a background process, so that your main thread can continue working.


To make this all possible, Pescador provides the following utilities:

    - :ref:`Streamer` objects encapsulate data generators for re-use, infinite sampling, and inter-process
      communication.
    - :ref:`Mux` objects allow flexible sampling from multiple streams
    - :ref:`ZMQStreamer` provides parallel processing with low communication overhead
    - Transform or modify streams with Maps (see :ref:`processing-data-streams`)
    - Buffering of sampled data into fixed-size batches (see :ref:`pescador.buffer_stream`)

************
Installation
************

Pescador can be installed from PyPI through `pip`:

.. code-block:: bash

    pip install pescador

or via `conda` using the `conda-forge` channel:

.. code-block:: bash

    conda install -c conda-forge pescador


************
Introduction
************
.. toctree::
    :maxdepth: 2

    intro

*************
Why Pescador?
*************
.. toctree::
    :maxdepth: 2

    why

**************
Basic examples
**************
.. toctree::
    :maxdepth: 2

    examples

*****************
Advanced examples
*****************
.. toctree::
    :maxdepth: 2

    auto_examples/index

*************
API Reference
*************
.. toctree::
    :maxdepth: 2

    api

*************
Release notes
*************
.. toctree::
    :maxdepth: 2

    changes

**********
Contribute
**********
- `Issue Tracker <http://github.com/pescadores/pescador/issues>`_
- `Source Code <http://github.com/pescadores/pescador>`_
- `Contributing guidelines <https://github.com/pescadores/pescador/blob/master/CONTRIBUTING.md>`_

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
