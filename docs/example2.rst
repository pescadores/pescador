.. _example2:

Advanced example
================

This document will walk through advanced usage of pescador.

We will assume a working understanding of the simple example in the previous section.


Streamers
---------
Generators in python have a couple of limitations for common stream learning pipelines.  First, once
instantiated, a generator cannot be "restarted".  Second, an instantiated generator cannot be serialized
directly, so they are difficult to use in distributed computation environments.

Pescador provides the `Streamer` object to circumvent these issues.  `Streamer` simply provides an object
container for an uninstantiated generator (and its parameters), and an access method `generate()`.  Calling
`generate()` multiple times on a streamer object is equivalent to restarting the generator, and can therefore
be used to simply implement multiple pass streams.  Similarly, because `Streamer` can be serialized, it is
simple to pass a streamer object to a separate process for parallel computation.

Here's a simple example, using the generator from the previous section.

.. code-block:: python
    :linenos:

    import pescador

    streamer = pescador.Streamer(noisy_samples, X[train], Y[train])

    batch_stream2 = streamer.generate()

Iterating over `streamer.generate()` is equivalent to iterating over `noisy_samples(X[train], Y[train])`.

Additionally, Streamer can be bounded easily by saying `streamer.generate(max_batches=N)` for some `N` maximum 
number of batches.


Stream re-use and multiplexing
------------------------------
The `mux()` function provides a powerful interface for randomly interleaving samples from 
multiple input streams.




Parallel streaming
------------------
