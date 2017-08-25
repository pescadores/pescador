.. _intro:

************
Introduction
************

Pescador's primary goal is to provide fine-grained control over data streaming and sampling.
These problems can get complex quickly, so this section provides an overview of the concepts underlying
Pescador's design, and a summary of the provided functionality.


Definitions
-----------

To understand what pescador does, it will help to establish some common terminology.
If you're not already familiar with Python's `iterator` and `generator` concepts, here's a quick synopsis:

1. An `iterator` is an object that produces a sequence of data, i.e. via ``__next__`` / ``next()``. 
   
   - See: `iterator definition <https://docs.python.org/3/glossary.html#term-iterator>`_, `Iterator Types <https://docs.python.org/3/library/stdtypes.html#typeiter>`_

2. An `iterable` is an object that can produce iterators, i.e. via ``__iter__`` / ``iter()``. 
   
   - See: `iterable definition <https://docs.python.org/3/glossary.html#term-iterable>`_

3. A `generator` (or more precisely `generator function`) is a callable object that returns a single iterator. 
   
   - See: `generator definition <https://docs.python.org/3/glossary.html#term-generator>`_

4. Pescador defines a `stream` as the sequence of objects produced by an iterator.


For example:
    - ``range`` is an iterable function
    - ``range(8)`` is an iterable, and its iterator produces the stream: ``0, 1, 2, 3, ...``


.. _streaming-data:

Streaming Data
--------------
1. Pescador defines an object called a `Streamer` for the purposes of (re)creating iterators indefinitely and (optionally) interrupting them prematurely.

2. `Streamer` implements the `iterable` interface, and can be iterated directly.

3. A `Streamer` can be initialized with one of two types:
    - Any iterable type, e.g. ``range(7)``, ``['foo', 'bar']``, ``"abcdef"``, or another ``Streamer``
    - A generator function and its arguments + keyword arguments.

4. A `Streamer` transparently yields the data stream flowing through it

    - A `Streamer` should not modify objects in its stream.

    - In the spirit of encapsulation, the modification of data streams is achieved through separate functionality (see :ref:`processing-data-streams`)


Multiplexing Data Streams
-------------------------
1. Pescador defines an object called a `Mux` for the purposes of multiplexing streams of data.

2. `Mux` inherits from `Streamer`, which makes it both iterable and recomposable.  Muxes allow you to
   construct arbitrary trees of data streams.  This is useful for hierarchical sampling.

3. A `Mux` is initialized with a container of one or more iterables, and parameters to control the stochastic behavior of the object.

4. As a subclass of `Streamer`, a `Mux` also transparently yields the stream flowing through it, i.e. :ref:`streaming-data`.


.. _processing-data-streams:

Processing Data Streams
-----------------------
Pescador adopts the concept of "transformers" for processing data streams.

1. A transformer takes as input a single object in the stream.

2. A transformer yields an object.

3. Transformers are iterators, i.e. implement a `__next__` method, to preserve iteration.

4. An example of a built-in transformer is `enumerate` [`ref <https://docs.python.org/3.3/library/functions.html#enumerate>`_]
