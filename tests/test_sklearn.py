#!/usr/bin/env python
'''Tests for sklearn integration'''

from __future__ import print_function

import six

import numpy as np

import sklearn
import sklearn.cluster
import sklearn.linear_model
import sklearn.datasets

import pescador

from nose.tools import raises


def generate_data(size=32, supervised=True):
    # Load up the iris dataset for the demo
    data = sklearn.datasets.load_iris()

    X, Y = data.data, data.target

    n, d = X.shape

    while True:
        i = np.random.randint(0, n, size=size)

        noise = np.random.randn(size, d)

        if supervised:
            yield {'X': X[i] + noise, 'Y': Y[i]}
        else:
            yield {'X': X[i] + noise}


def test_unsupervised():

    def __test(sup, max_steps):

        stream = pescador.Streamer(generate_data, supervised=sup)

        estimator = sklearn.cluster.MiniBatchKMeans()

        model = pescador.StreamLearner(estimator, max_steps=max_steps)

        model.iter_fit(stream.generate(max_batches=100))

    for sup in [False, True]:
        for max_steps in [-1, None, 1]:
            if not sup and max_steps != -1:
                yield __test, sup, max_steps
            else:
                yield (raises(RuntimeError, AssertionError)(__test),
                       sup, max_steps)


def test_supervised():

    def __test(sup, max_steps):

        stream = pescador.Streamer(generate_data, supervised=sup)

        estimator = sklearn.linear_model.SGDClassifier()

        model = pescador.StreamLearner(estimator, max_steps=max_steps)

        model.iter_fit(stream.generate(max_batches=100),
                       classes=[0, 1, 2])

    for sup in [False, True]:
        for max_steps in [-1, None, 1]:
            if sup and max_steps != -1:
                yield __test, sup, max_steps
            else:
                yield (raises(RuntimeError, AssertionError)(__test),
                       sup, max_steps)
