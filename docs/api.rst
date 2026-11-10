.. _api:

API Reference
=============

.. module:: pescador


.. _Streamer:
.. _PrefetchStreamer:
.. _HybridBuffer:
.. _ZMQStreamer:

Streaming
---------

.. currentmodule:: pescador

.. autosummary::
    :toctree: generated/

    Streamer
    PrefetchStreamer
    HybridBuffer
    ZMQStreamer
    streamable


.. _Mux:

Multiplexing
------------

.. currentmodule:: pescador

.. autosummary::
    :toctree: generated/

    StochasticMux
    ShuffledMux
    RoundRobinMux
    ChainMux
    BaseMux


.. _Maps:

Maps (Transformers)
-------------------

.. currentmodule:: pescador

.. autosummary::
    :toctree: generated/

    buffer_stream
    tuples
    keras_tuples
    cache


.. _Exceptions:

Exceptions
----------

.. currentmodule:: pescador

.. autosummary::
    :toctree: generated/

    PescadorError
    DataError
