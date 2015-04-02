#!/usr/bin/env python
'''Test the streamer object for reusable generators'''

import itertools
import six

import numpy as np

import pescador

from nose.tools import raises, eq_


def __eq_batch(b1, b2):

    for k in six.iterkeys(b1):
        assert np.allclose(b1[k], b2[k])


def finite_generator(n, size=2):

    for i in range(n):
        yield {'X': np.tile(np.array([[i]]), (size, 1))}


def infinite_generator(size=2):

    i = 0
    while True:
        yield {'X': np.tile(np.array([[i]]), (size, 1))}
        i = i + 1


def test_streamer_list():

    reference = list(finite_generator(10))

    query = list(pescador.Streamer(reference).generate())

    for b1, b2 in zip(reference, query):
        __eq_batch(b1, b2)


def test_streamer_finite():

    def __test(n_max, size):
        reference = list(finite_generator(50, size=size))

        if n_max is not None:
            reference = reference[:n_max]

        streamer = pescador.Streamer(finite_generator, 50, size=size)

        for i in range(3):
            query = list(streamer.generate(max_batches=n_max))
            for b1, b2 in zip(reference, query):
                __eq_batch(b1, b2)

    for n_max in [None, 10, 50, 100]:
        for size in [1, 2, 7]:
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
            query = list(streamer.generate(max_batches=n_max))

            for b1, b2 in zip(reference, query):
                __eq_batch(b1, b2)

    for n_max in [10, 50]:
        for size in [1, 2, 7]:
            yield __test, n_max, size


@raises(TypeError)
def test_streamer_bad_function():

    def __fail():
        return 5

    _ = pescador.Streamer(__fail)


def test_zmq():

    stream = pescador.Streamer(finite_generator, 20, size=3)

    reference = list(stream.generate())

    for i in range(3):
        query = list(pescador.zmq_stream(5155, stream))
        for b1, b2 in zip(reference, query):
            __eq_batch(b1, b2)


def __zip_generator(n, size1, size2):

    for b1, b2 in itertools.izip(finite_generator(n, size=size1),
                                 finite_generator(n, size=size2)):
        yield dict(X=b1['X'], Y=b2['X'])


def test_batch_length():
    def __test(generator, n):
        for batch in generator:
            eq_(pescador.util.batch_length(batch), n)

    for n1 in [5, 10, 15]:
        for n2 in [5, 10, 15]:
            if n1 != n2:
                test = raises(RuntimeError)(__test)
            else:
                test = __test
            yield test, __zip_generator(3, n1, n2), n1
