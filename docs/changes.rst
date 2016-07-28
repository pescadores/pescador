Changes
=======

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
