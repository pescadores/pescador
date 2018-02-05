Changes
=======

v2.0.0 (2018-02-05)
-------------------
This release is the second major revision of the pescador architecture, and
includes many substantial changes to the API.

This release contains no changes from the release candidate 2.0.0rc0.

- `#103`_ Deprecation and refactor of the `Mux` class.  Its functionality has 
  been superseded by new classes `StochasticMux`, `ShuffledMux`, `ChainMux`,
  and `RoundRobinMux`.
- `#109`_ Removed deprecated features from the 1.x series: 
  - `BufferedStreamer` class
  - `Streamer.tuples` method
- `#111`_ Removed the internally-facing `StreamActivator` class
- `#113`_ Bugfix: multiply-activated streamers (and muxes) no longer share state
- `#116`_ `Streamer.cycle` now respects the `max_iter` parameter
- `#121`_ Added minimum dependency version requirements
- `#106`_ Added more advanced examples in the documentation

.. _#103: https://github.com/pescadores/pescador/pull/103
.. _#109: https://github.com/pescadores/pescador/pull/109
.. _#111: https://github.com/pescadores/pescador/pull/111
.. _#113: https://github.com/pescadores/pescador/pull/113
.. _#116: https://github.com/pescadores/pescador/pull/116
.. _#121: https://github.com/pescadores/pescador/pull/121
.. _#106: https://github.com/pescadores/pescador/pull/106

v1.1.0 (2017-08-25)
-------------------
This is primarily a maintenance release, and will be the last in the 1.x series.

- `#97`_ Fixed an infinite loop in `Mux`
- `#91`_ Changed the default timeout for `ZMQStreamer` to 5 seconds.
- `#90`_ Fixed conda-forge package distribution
- `#89`_ Refactored internals of the `Mux` class toward the 2.x series
- `#88`_, `#100`_ improved unit tests
- `#73`_, `#95`_ Updated documentation

.. _#73: https://github.com/pescadores/pescador/pull/73
.. _#88: https://github.com/pescadores/pescador/pull/88
.. _#89: https://github.com/pescadores/pescador/pull/89
.. _#90: https://github.com/pescadores/pescador/pull/90
.. _#91: https://github.com/pescadores/pescador/pull/91
.. _#95: https://github.com/pescadores/pescador/pull/95
.. _#97: https://github.com/pescadores/pescador/pull/97
.. _#100: https://github.com/pescadores/pescador/pull/100

v1.0.0 (2017-03-18)
-------------------
This release constitutes a major revision over the 0.x series, and the new interface
is not backward-compatible.

- `#23`_ Preserve memory alignment of numpy arrays across ZMQ streams
- `#34`_ Rewrite of all core functionality under a unified object interface ``Streamer``.
- `#35`_, `#52`_ Removed the `StreamLearner` interface and scikit-learn dependency
- `#44`_ Added a base exception class `PescadorError`
- `#45`_ Removed the `util` submodule
- `#47`_, `#60`_ Improvements to documentation
- `#48`_ Fixed a timeout bug in ZMQ streamer
- `#53`_ Added testing and support for python 3.6
- `#57`_ Added the `.tuple()` interface for Streamers
- `#61`_ Improved test coverage
- `#63`_ Added random state to `Mux`
- `#64`_ Added `__call__` interface to `Streamer`


.. _#64: https://github.com/pescadores/pescador/pull/64
.. _#63: https://github.com/pescadores/pescador/pull/63
.. _#61: https://github.com/pescadores/pescador/pull/61
.. _#57: https://github.com/pescadores/pescador/pull/57
.. _#53: https://github.com/pescadores/pescador/pull/53
.. _#48: https://github.com/pescadores/pescador/pull/48
.. _#60: https://github.com/pescadores/pescador/pull/60
.. _#47: https://github.com/pescadores/pescador/pull/47
.. _#45: https://github.com/pescadores/pescador/pull/45
.. _#44: https://github.com/pescadores/pescador/pull/44
.. _#52: https://github.com/pescadores/pescador/pull/52
.. _#35: https://github.com/pescadores/pescador/pull/35
.. _#34: https://github.com/pescadores/pescador/pull/34
.. _#23: https://github.com/pescadores/pescador/pull/23

v0.1.3 (2016-07-28)
-------------------
- Added support for ``joblib>=0.10``

v0.1.2 (2016-02-22)
-------------------

- Added ``pescador.mux`` parameter `revive`.  Calling with `with_replacement=False, revive=True`
  will use each seed at most once at any given time.
- Added ``pescador.zmq_stream`` parameter `timeout`. Setting this to a positive number will terminate
  dangling worker threads after `timeout` is exceeded on join.  See also: ``multiprocessing.Process.join``.

v0.1.1 (2016-01-07)
-------------------

- ``pescador.mux`` now throws a ``RuntimeError`` exception if the seed pool is empty


v0.1.0 (2015-10-20)
-------------------
Initial public release
