.. _example2:

This example demonstrates how to re-use and multiplex streamers.

We will assume a working understanding of the simple example in the previous section.

Stream re-use and multiplexing
==============================

The `StochasticMux` streamer provides a powerful interface for randomly interleaving samples from multiple input streams.
`StochasticMux` can also dynamically activate and deactivate individual `Streamers`, which allows it to operate on a bounded subset of streams at any given time.

As a concrete example, we can simulate a mixture of noisy streams with differing variances.

.. code-block:: python
    :linenos:

    import numpy as np

    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import ShuffleSplit
    from sklearn.metrics import accuracy_score

    from pescador import Streamer, StochasticMux

    def noisy_samples(X, Y, sigma=1.0):
        '''Copied over from the previous example'''
        n, d = X.shape

        while True:
            i = np.random.randint(0, n, size=1)

            noise = sigma * np.random.randn(1, d)

            yield dict(X=X[i] + noise, Y=Y[i])

    # Load some example data from sklearn
    raw_data = load_breast_cancer()
    X, Y = raw_data['data'], raw_data['target']

    classes = np.unique(Y)

    rs = ShuffleSplit(n_splits=1, test_size=0.1)
    for train, test in rs.split(X):

        # Instantiate a linear classifier
        estimator = SGDClassifier()

        # Build a collection of Streamers with different noise scales
        streams = [Streamer(noisy_samples, X[train], Y[train], sigma=sigma)
                   for sigma in [0, 0.5, 1.0, 2.0, 4.0]]

        # Build a mux stream, keeping 3 streams alive at once
        mux_stream = StochasticMux(streams,
                                   3,        # Keep 3 streams alive at once
                                   rate=64)  # Use a poisson rate of 64

        # Fit the model to the stream, use at most 5000 samples
        for sample in mux_stream(max_iter=5000):
            estimator.partial_fit(sample['X'], sample['Y'], classes=classes)

        # And report the accuracy
        Ypred = estimator.predict(X[test])
        print('Test accuracy: {:.3f}'.format(accuracy_score(Y[test], Ypred)))


In the above example, each `Streamer` in `streams` can make infinitely many samples. The `rate=64` argument to
`StochasticMux` says that each stream should produce some `n` samples, where `n` is sampled from a Poisson distribution of rate `rate`.
When a stream exceeds its bound, it is deactivated, and a new streamer is activated to fill its place.

Setting `rate=None` disables the random stream bounding, and simply runs each active stream until exhaustion.

The `StochasticMux` streamer can sampled with or without replacement from its input streams, according to the `mode` option.
Setting this parameter to `single_active` means that each stream can be active at most once.

Streams can also be sampled with non-uniform weighting by specifying a vector of `weights`.

Finally, exhausted streams can be removed by setting `mode='exhaustive'`.
