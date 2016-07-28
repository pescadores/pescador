.. _example1:

Simple example
==============

This document will walk through the basics of training models using pescador.

Our running example will be learning from an infinite stream of stochastically perturbed samples
from the Iris dataset.

Before we can get started, we'll need to introduce a few core concepts.  We will assume some basic
familiarity with `scikit-learn <http://scikit-learn.org/stable/>`_ and 
`generators <https://wiki.python.org/moin/Generators>`_.


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
        batch
        '''


        n, d = X.shape

        while True:
            i = np.random.randint(0, n, size=m)

            noise = sigma * np.random.randn(batch_size, d)

            yield dict(X=X[i] + noise, Y=Y[i])


In the code above, `noisy_samples` is a generator that can be sampled indefinitely because `noisy_samples`
contains an infinite loop.  Each iterate of `noisy_samples` will be a dictionary containing the sample batch's
features and labels.


StreamLearner
-------------

Many scikit-learn classes provide an iterative learning interface via `partial_fit()`, which can update an
existing model after observing a new batch of samples.  Pescador provides an additional layer
(`StreamLearner`) which interfaces between batch generators and `partial_fit()`.

The following example illustrates how to use `StreamLearner`.

.. code-block:: python
    :linenos:

    from __future__ import print_function

    import sklearn.datasets
    from sklearn.cross_validation import ShuffleSplit
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score

    import pescador

    # Load the Iris dataset
    data = sklearn.datasets.load_iris()
    X, Y = data.data, data.target

    # Get the space of class labels
    classes = np.unique(Y)

    # Generate a single 90/10 train/test split
    for train, test in ShuffleSplit(len(X), n_iter=1, test_size=0.1)

        # Instantiate a linear classifier
        estimator = SGDClassifier()

        # Wrap the estimator object in a stream learner
        model = pescador.StreamLearner(estimator, max_batches=1000)

        # Build a data stream
        batch_stream = noisy_samples(X[train], Y[train])

        # Fit the model to the stream
        model.iter_fit(batch_stream, classes=classes)

        # And report the accuracy
        print('Test accuracy: {:.3f}'.format(accuracy_score(Y[test],
                                                            model.predict(X[test]))))

A few things to note here:

    * Because `noisy_samples` is an infinite generator, we need to provide an explicit bound on the amount of
      samples to draw when fitting.  This is done in line 20 with the `max_batches` parameter to
      `StreamLearner`.


    * `StreamLearner` objects transparently wrap the methods of their contained `estimator` object, so
      `model.predict(X[test])` and `model.estimator.predict(X[test])` are equivalent.
