.. _example1:

Basic example
==============

This document will walk through the basics of using pescador to stream samples from a generator.

Our running example will be learning from an infinite stream of stochastically perturbed samples from the Iris dataset.

Before we can get started, we'll need to introduce a few core concepts.  
We will assume some basic familiarity with `generators <https://wiki.python.org/moin/Generators>`_.


Batch generators
----------------
Not all python generators are valid for machine learning.  Pescador assumes that generators produce output in
a particular format, which we will refer to as a `batch`.  Specifically, a batch is a python dictionary
containing `np.ndarray`.  For unsupervised learning (e.g., MiniBatchKMeans), valid batches contain only one
key: `X`.  For supervised learning (e.g., SGDClassifier), valid batches must contain both `X` and `Y` keys,
both of equal length.

Here's a simple example generator that draws random batches of data from Iris of a specified `batch_size`,
and adds gaussian noise to the features.

.. code-block:: python
    :linenos:

    import numpy as np

    def noisy_samples(X, Y, batch_size=16, sigma=1.0):
        '''Generate an infinite stream of noisy samples from a labeled dataset.
        
        Parameters
        ----------
        X : np.ndarray, shape=(n, d)
            Features

        Y : np.ndarray, shape=(n,)
            Labels

        batch_size : int > 0
            Size of the batches to generate

        sigma : float > 0
            Variance of the additive noise

        Yields
        ------
        batch : dict
            batch['X'] is an `np.ndarray` of shape `(batch_size, d)`

            batch[Y'] is an `np.ndarray` of shape `(batch_size,)`
        '''


        n, d = X.shape

        while True:
            i = np.random.randint(0, n, size=batch_size)

            noise = sigma * np.random.randn(batch_size, d)

            yield dict(X=X[i] + noise, Y=Y[i])


In the code above, `noisy_samples` is a generator that can be sampled indefinitely because `noisy_samples`
contains an infinite loop.  Each iterate of `noisy_samples` will be a dictionary containing the sample batch's
features and labels.


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

Additionally, Streamer can be bounded easily by saying `streamer.generate(max_batches=N)` for some `N` maximum number of batches.
