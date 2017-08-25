.. _example1:

Streaming data
==============

This example will walk through the basics of using pescador to stream samples from a generator.

Our running example will be learning from an infinite stream of stochastically perturbed samples from the Iris dataset.


Sample generators
-----------------
Streamers are intended to transparently pass data without modifying them.
However, Pescador assumes that Streamers produce output in a particular format.
Specifically, a data is expected to be a python dictionary where each value contains a `np.ndarray`.
For an unsupervised learning (e.g., SKLearn/`MiniBatchKMeans`), the data might contain only one key: `X`.
For supervised learning (e.g., SGDClassifier), valid data would contain both `X` and `Y` keys, both of equal length.

Here's a simple example generator that draws random samples of data from the Iris dataset, and adds gaussian noise to the features.

.. code-block:: python
    :linenos:

    import numpy as np

    def noisy_samples(X, Y, sigma=1.0):
        '''Generate an infinite stream of noisy samples from a labeled dataset.
        
        Parameters
        ----------
        X : np.ndarray, shape=(d,)
            Features

        Y : np.ndarray, shape=(,)
            Labels

        sigma : float > 0
            Variance of the additive noise

        Yields
        ------
        sample : dict
            sample['X'] is an `np.ndarray` of shape `(d,)`

            sample['Y'] is a scalar `np.ndarray` of shape `(,)`
        '''

        n, d = X.shape

        while True:
            i = np.random.randint(0, n)

            noise = sigma * np.random.randn(1, d)

            yield dict(X=X[i] + noise, Y=Y[i])

In the code above, `noisy_samples` is a generator that can be sampled indefinitely because `noisy_samples` contains an infinite loop.
Each iterate of `noisy_samples` will be a dictionary containing the sample's features and labels.


Streamers
---------
Generators in python have a couple of limitations for common stream learning pipelines.
First, once instantiated, a generator cannot be "restarted".
Second, an instantiated generator cannot be serialized directly, so they are difficult to use in distributed computation environments.

Pescador provides the `Streamer` class to circumvent these issues.
`Streamer` simply provides an object container for an uninstantiated generator (and its parameters), and an access method `generate()`.
Calling `generate()` multiple times on a `Streamer` object is equivalent to restarting the generator, and can therefore be used to simply implement multiple pass streams.
Similarly, because `Streamer` can be serialized, it is simple to pass a streamer object to a separate process for parallel computation.

Here's a simple example, using the generator from the previous section.

.. code-block:: python
    :linenos:

    import pescador

    streamer = pescador.Streamer(noisy_samples, X[train], Y[train])

    stream2 = streamer.iterate()

Iterating over `streamer.iterate()` is equivalent to iterating over `noisy_samples(X[train], Y[train])`.

Additionally, Streamer can be bounded easily by saying `streamer.iterate(max_iter=N)` for some `N` maximum number of samples.

Finally, because `iterate()` is such a common operation with streamer objects, a short-hand interface is provided by treating the streamer object as if it was a generator:

.. code-block:: python
    :linenos:

    import pescador

    streamer = pescador.Streamer(noisy_samples, X[train], Y[train])

    # Equivalent to stream2 above
    stream3 = streamer()


Iterating over any of these would then look like the following:

.. code-block:: python
    :linenos:

    for sample in streamer.iterate():
        # do something
        ...

    # For convenience, the object directly behaves as an iterator.
    for sample in streamer:
        # do something
        ...
