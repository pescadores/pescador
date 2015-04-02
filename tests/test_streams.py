#!/usr/bin/env python
'''Test the streamer object for reusable generators'''
from __future__ import print_function

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
        for i, data in enumerate(infinite_generator(size=size)):
            if i >= n_max:
                break
            reference.append(data)

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
        return 6

    pescador.Streamer(__fail)


def test_zmq():

    stream = pescador.Streamer(finite_generator, 20, size=3)

    reference = list(stream.generate())

    for i in range(3):
        query = list(pescador.zmq_stream(5155, stream))
        for b1, b2 in zip(reference, query):
            __eq_batch(b1, b2)


def __zip_generator(n, size1, size2):

    for b1, b2 in zip(finite_generator(n, size=size1),
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


def test_buffer_batch():

    def __serialize_batches(batches):

        for batch in batches:
            for item in batch['X']:
                yield item

    def __test(n_batch, n_buf):
        reference = finite_generator(50, size=n_batch)

        reference = __serialize_batches(reference)

        estimate = pescador.buffer_batch(finite_generator(50, size=n_batch),
                                         n_buf)
        estimate = __serialize_batches(estimate)

        eq_(list(reference), list(estimate))

    for batch_size in [1, 2, 5, 17]:
        for buf_size in [1, 2, 5, 17, 100]:
            yield __test, batch_size, buf_size


def test_mux_single():

    reference = list(finite_generator(50))
    stream = pescador.Streamer(reference)

    estimate = pescador.mux([stream], None, 1, with_replacement=False)
    eq_(list(reference), list(estimate))


def test_mux_weighted():

    def __test(weight):
        reference = list(finite_generator(50))
        noise = list(finite_generator(50, size=1))
        stream = pescador.Streamer(reference)
        stream2 = pescador.Streamer(noise)
        estimate = pescador.mux([stream, stream2], None, 2,
                                pool_weights=[1.0, weight],
                                with_replacement=False)
        eq_(list(reference), list(estimate))

    yield __test, 0.0
    yield raises(AssertionError)(__test), 0.5


def test_mux_rare():

    def __test(weight):
        reference = list(finite_generator(50))
        noise = list(finite_generator(50, size=1))
        stream = pescador.Streamer(reference)
        stream2 = pescador.Streamer(noise)
        estimate = pescador.mux([stream, stream2], None, 2,
                                pool_weights=weight,
                                with_replacement=False)
        eq_(list(reference) + list(noise), list(estimate))

    # This should give us all the reference before all the noise
    yield __test, [1e10, 1e-10]


def test_empty_seeds():

    def __empty():
        if False:
            yield 1

    reference = pescador.Streamer(finite_generator, 10)
    empty = pescador.Streamer(__empty)

    estimate = pescador.mux([reference, empty], 10, 2, lam=None,
                            with_replacement=False,
                            pool_weights=[1e-10, 1e10])

    estimate = list(estimate)

    ref = list(reference.generate())
    for b1, b2 in zip(ref, estimate):
        __eq_batch(b1, b2)


def test_mux_replacement():

    def __test(n_streams, n_samples, k, lam):

        seeds = [pescador.Streamer(infinite_generator)
                 for _ in range(n_streams)]

        mux = pescador.mux(seeds, n_samples, k, lam=lam)

        estimate = list(mux)

        # Make sure we get the right number of samples
        eq_(len(estimate), n_samples)

    for n_streams in [1, 2, 4]:
        for n_samples in [10, 20, 80]:
            for k in [1, 2, 4]:
                for lam in [1.0, 2.0, 8.0]:
                    yield __test, n_streams, n_samples, k, lam
