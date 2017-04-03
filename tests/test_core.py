#!/usr/bin/env python
'''Test the streamer object for reusable generators'''
from __future__ import print_function
import pytest

import warnings
warnings.simplefilter('always')

import pescador.core
import test_utils as T


def test_StreamActivator():
    with pytest.raises(AttributeError):
        pescador.core.StreamActivator(range)


def test_streamer_list():

    reference = list(T.finite_generator(10))

    query = list(pescador.core.Streamer(reference, 10).generate())

    assert len(reference) == len(query)
    for b1, b2 in zip(reference, query):
        T.__eq_batch(b1, b2)


@pytest.mark.parametrize('items',
                         [['X'], ['Y'], ['X', 'Y'], ['Y', 'X'],
                          pytest.mark.xfail([],
                                            raises=pescador.PescadorError)])
def test_streamer_tuple(items):

    reference = [tuple(batch[it] for it in items)
                 for batch in T.__zip_generator(10, 2, 3)]

    streamer = pescador.core.Streamer(T.__zip_generator, 10, 2, 3)
    query = list(streamer.tuples(*items))

    assert len(reference) == len(query)
    for b1, b2 in zip(reference, query):
        assert isinstance(b2, tuple)
        T.__eq_lists(b1, b2)


@pytest.mark.parametrize('n_max', [None, 10, 50, 100])
@pytest.mark.parametrize('stream_size', [1, 2, 7])
@pytest.mark.parametrize('generate', [False, True])
def test_streamer_finite(n_max, stream_size, generate):
    reference = list(T.finite_generator(50, size=stream_size))

    if n_max is not None:
        reference = reference[:n_max]

    streamer = pescador.core.Streamer(T.finite_generator, 50, size=stream_size)

    if generate:
        gen = streamer.generate(max_batches=n_max)
    else:
        gen = streamer(max_batches=n_max)

    for i in range(3):

        query = list(gen)
        for b1, b2 in zip(reference, query):
            T.__eq_batch(b1, b2)


@pytest.mark.parametrize('n_max', [10, 50])
@pytest.mark.parametrize('stream_size', [1, 2, 7])
def test_streamer_infinite(n_max, stream_size):
    reference = []
    for i, data in enumerate(T.infinite_generator(size=stream_size)):
        if i >= n_max:
            break
        reference.append(data)

    streamer = pescador.core.Streamer(T.infinite_generator, size=stream_size)

    for i in range(3):
        query = list(streamer.generate(max_batches=n_max))

        for b1, b2 in zip(reference, query):
            T.__eq_batch(b1, b2)


@pytest.mark.parametrize('n_max', [10, 50])
@pytest.mark.parametrize('stream_size', [1, 2, 7])
def test_streamer_in_streamer(n_max, stream_size):
    # TODO minimize copypasta from above test.
    reference = []
    for i, data in enumerate(T.infinite_generator(size=stream_size)):
        if i >= n_max:
            break
        reference.append(data)

    streamer = pescador.core.Streamer(T.infinite_generator, size=stream_size)

    streamer2 = pescador.core.Streamer(streamer)

    for i in range(3):
        query = list(streamer2.generate(max_batches=n_max))

        for b1, b2 in zip(reference, query):
            T.__eq_batch(b1, b2)


@pytest.mark.parametrize('generate', [False, True])
def test_streamer_cycle(generate):
    """Test that a limited streamer will die and restart automatically."""
    stream_len = 10
    streamer = pescador.core.Streamer(T.finite_generator, stream_len)
    assert streamer.stream_ is None

    # Exhaust the stream once.
    query = list(streamer.generate())
    assert stream_len == len(query)

    # Now, generate from it infinitely using cycle.
    # We're going to assume "infinite" == > 5*stream_len
    count_max = 5 * stream_len

    data_results = []
    if generate:
        gen = streamer.cycle()
    else:
        gen = streamer(cycle=True)

    for i, x in enumerate(gen):
        data_results.append((isinstance(x, dict) and 'X' in x))
        if (i + 1) >= count_max:
            break
    assert (len(data_results) == count_max and all(data_results))


@pytest.mark.parametrize(
    'items',
    [['X'], ['Y'], ['X', 'Y'], ['Y', 'X'],
     pytest.mark.xfail([], raises=pescador.core.PescadorError)])
def test_streamer_cycle_tuples(items):
    """Test that a limited streamer will die and restart automatically."""
    stream_len = 10
    streamer = pescador.core.Streamer(T.__zip_generator, 10, 2, 3)
    assert streamer.stream_ is None

    # Exhaust the stream once.
    query = list(streamer.generate())
    assert stream_len == len(query)

    # Now, generate from it infinitely using cycle.
    # We're going to assume "infinite" == > 5*stream_len
    count_max = 5 * stream_len

    data_results = []
    kwargs = dict(cycle=True)
    for i, x in enumerate(streamer.tuples(*items, **kwargs)):
        data_results.append((isinstance(x, tuple)))
        if (i + 1) >= count_max:
            break
    assert (len(data_results) == count_max and all(data_results))


def test_streamer_bad_function():

    def __fail():
        return 6

    with pytest.raises(pescador.core.PescadorError):
        pescador.Streamer(__fail)
