#!/usr/bin/env python
'''Test the streamer object for reusable generators'''
from __future__ import print_function

from nose.tools import raises, eq_
import warnings
warnings.simplefilter('always')

import pescador
import test_utils as T


def test_streamer_list():

    reference = list(T.finite_generator(10))

    query = list(pescador.Streamer(reference, 10).generate())

    eq_(len(reference), len(query))
    for b1, b2 in zip(reference, query):
        T.__eq_batch(b1, b2)


def test_streamer_finite():

    def __test(n_max, size):
        reference = list(T.finite_generator(50, size=size))

        if n_max is not None:
            reference = reference[:n_max]

        streamer = pescador.Streamer(T.finite_generator, 50, size=size)

        for i in range(3):
            query = list(streamer.generate(max_batches=n_max))
            for b1, b2 in zip(reference, query):
                T.__eq_batch(b1, b2)

    for n_max in [None, 10, 50, 100]:
        for size in [1, 2, 7]:
            yield __test, n_max, size


def test_streamer_infinite():

    def __test(n_max, size):
        reference = []
        for i, data in enumerate(T.infinite_generator(size=size)):
            if i >= n_max:
                break
            reference.append(data)

        streamer = pescador.Streamer(T.infinite_generator, size=size)

        for i in range(3):
            query = list(streamer.generate(max_batches=n_max))

            for b1, b2 in zip(reference, query):
                T.__eq_batch(b1, b2)

    for n_max in [10, 50]:
        for size in [1, 2, 7]:
            yield __test, n_max, size


def test_streamer_in_streamer():
    # TODO minimize copypasta from above test.

    def __test(n_max, size):
        reference = []
        for i, data in enumerate(T.infinite_generator(size=size)):
            if i >= n_max:
                break
            reference.append(data)

        streamer = pescador.Streamer(T.infinite_generator, size=size)

        streamer2 = pescador.Streamer(streamer)

        for i in range(3):
            query = list(streamer2.generate(max_batches=n_max))

            for b1, b2 in zip(reference, query):
                T.__eq_batch(b1, b2)

    for n_max in [10, 50]:
        for size in [1, 2, 7]:
            yield __test, n_max, size


def test_streamer_cycle():
    """Test that a limited streamer will die and restart automatically."""

    stream_len = 10
    streamer = pescador.Streamer(T.finite_generator, stream_len)
    assert streamer.stream_ is None

    # Exhaust the stream once.
    query = list(streamer.generate())
    eq_(stream_len, len(query))

    # Now, generate from it infinitely using cycle.
    # We're going to assume "infinite" == > 5*stream_len
    count_max = 5 * stream_len
    success = False
    for i, x in enumerate(streamer.cycle()):
        if i >= count_max:
            success = True
            break
    assert success


@raises(TypeError)
def test_streamer_bad_function():

    def __fail():
        return 6

    pescador.Streamer(__fail)
