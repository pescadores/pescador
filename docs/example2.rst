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
The `Mux` streamer provides a powerful interface for randomly interleaving samples from
multiple input streams.  `Mux` can also dynamically activate and deactivate
individual `Streamers`, which allows it to operate on a bounded subset of streams at any given time.

As a concrete example, we can simulate a mixture of noisy streams with differing variances.

.. code-block:: python
    :linenos:

    for train, test in ShuffleSplit(len(X), n_iter=1, test_size=0.1)

        # Instantiate a linear classifier
        estimator = SGDClassifier()

        # Build a collection of streams with different variance scales
        streams = [noisy_samples(X[train], Y[train], sigma=sigma)
                   for sigma in [0.5, 1.0, 2.0, 4.0]]

        # Build a mux stream, keeping only 2 streams alive at once
        batch_stream = pescador.Mux(streams,
                                    1000,   # Generate 1000 batches in total
                                    2,      # Keep 2 streams alive at once
                                    lam=16) # Use a poisson rate of 16


        # Fit the model to the stream
        for batch in batch_stream:
            estimator.partial_fit(batch_stream, classes=classes)

        # And report the accuracy
        print('Test accuracy: {:.3f}'.format(accuracy_score(Y[test],
                                                            estimator.predict(X[test]))))

In the above example, each `noisy_samples` streamer is infinite.  The `lam=16` argument to `mux` 
says that each stream should produce some `n` batches, where `n` is sampled from a Poisson distribution
of rate `lam`.  When a stream exceeds its bound, it is deactivated, and a new stream is activated to fill its
place.

Setting `lam=None` disables the random stream bounding, and `mux()` simply runs each active stream until
exhaustion.

Streams can be sampled with or without replacement according to the `with_replacement` option.  Setting this
parameter to `False` means that each stream can be active at most once.

Streams can also be sampled with non-uniform weighting by specifying a vector `pool_weights`.

Finally, exhausted streams can be removed by setting `prune_empty_seeds` to `True`.  If `False`, then
exhausted streams may be reactivated at any time.


Note that because `Mux` itself is a Streamer, it too can be wrapped in a `Streamer` object.
