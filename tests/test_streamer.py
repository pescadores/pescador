#!/usr/bin/env python
'''Test the streamer object for reusable generators'''

import numpy as np
import pescador
import six


def __eq_batch(b1, b2):

    for k in six.iterkeys(b1):
        assert np.allclose(b1[k], b2[k])


def finite_generator(n, size=2):

    x = np.zeros((size, 3))
    for i in range(n):
        yield {'X': x.copy()}
        x[:] += i


def infinite_generator(p=31, size=2):

    x = np.zeros((size, 3))
    while True:
        yield {'X': x.copy()}
        x[:] = np.mod(x + 1, p)


def test_streamer_finite():

    def __test(n_max, size):
        reference = list(finite_generator(50, size=size))

        if n_max is not None:
            reference = reference[:n_max]

        streamer = pescador.Streamer(finite_generator, 50, size=size)

        for i in range(3):
            query = list(streamer.generate(max_items=n_max))
            for b1, b2 in zip(reference, query):
                __eq_batch(b1, b2)

    for n_max in [None, 10, 50, 100]:
        for size in [1, 2, 5]:
            yield __test, n_max, size


def test_streamer_infinite():

    def __test(n_max, size):
        reference = []
        for i in enumerate(infinite_generator(size=size)):
            if i >= n_max:
                break
            reference.append(i)

        streamer = pescador.Streamer(infinite_generator, size=size)

        for i in range(3):
            query = list(streamer.generate(max_items=n_max))

            for b1, b2 in zip(reference, query):
                __eq_batch(b1, b2)

    for n_max in [10, 50]:
        for size in [1, 2, 5]:
            yield __test, n_max, size

