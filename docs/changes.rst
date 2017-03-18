Changes
=======

v1.0.0
------
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

v0.1.3
------
- Added support for ``joblib>=0.10``

v0.1.2
------

- Added ``pescador.mux`` parameter `revive`.  Calling with `with_replacement=False, revive=True`
  will use each seed at most once at any given time.
- Added ``pescador.zmq_stream`` parameter `timeout`. Setting this to a positive number will terminate
  dangling worker threads after `timeout` is exceeded on join.  See also: ``multiprocessing.Process.join``.

v0.1.1
------

- ``pescador.mux`` now throws a ``RuntimeError`` exception if the seed pool is empty


v0.1.0
------
Initial public release
