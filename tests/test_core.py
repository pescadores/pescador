#!/usr/bin/env python
'''Test the streamer object for reusable iterators'''
import copy
import pytest

import warnings
warnings.simplefilter('always')

import pescador.core
import test_utils as T


def test_streamer_iterable():
    n_items = 10
    expected = list(range(n_items))
    streamer = pescador.core.Streamer(expected)

    # Test generate interface
    actual1 = list(streamer)
    assert len(expected) == len(actual1) == n_items
    for b1, b2 in zip(expected, actual1):
        assert b1 == b2

    # Test __iter__ interface
    actual2 = list(streamer)
    assert len(expected) == len(actual2) == n_items
    for b1, b2 in zip(expected, actual2):
        assert b1 == b2


def test_streamer_generator_func():
    n_items = 10
    expected = list(T.finite_generator(n_items))
    streamer = pescador.core.Streamer(T.finite_generator, n_items)

    # Test generate interface
    actual1 = list(streamer)
    assert len(expected) == len(actual1) == n_items
    for b1, b2 in zip(expected, actual1):
        T._eq_batch(b1, b2)

    # Test __iter__ interface
    actual2 = list(streamer)
    assert len(expected) == len(actual2) == n_items
    for b1, b2 in zip(expected, actual2):
        T._eq_batch(b1, b2)


@pytest.mark.parametrize('n_max', [None, 10, 50, 100])
@pytest.mark.parametrize('stream_size', [1, 2, 7])
@pytest.mark.parametrize('generate', [False, True])
def test_streamer_finite(n_max, stream_size, generate):
    reference = list(T.finite_generator(50, size=stream_size))

    if n_max is not None:
        reference = reference[:n_max]

    streamer = pescador.core.Streamer(T.finite_generator, 50, size=stream_size)

    if generate:
        gen = streamer.iterate(max_iter=n_max)
    else:
        gen = streamer(max_iter=n_max)

    for i in range(3):

        query = list(gen)
        for b1, b2 in zip(reference, query):
            T._eq_batch(b1, b2)


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
        query = list(streamer.iterate(max_iter=n_max))

        for b1, b2 in zip(reference, query):
            T._eq_batch(b1, b2)


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
        query = list(streamer2.iterate(max_iter=n_max))

        for b1, b2 in zip(reference, query):
            T._eq_batch(b1, b2)


@pytest.mark.parametrize('generate', [False, True])
def test_streamer_cycle(generate):
    """Test that a limited streamer will die and restart automatically."""
    stream_len = 10
    streamer = pescador.core.Streamer(T.finite_generator, stream_len)
    assert streamer.stream_ is None

    # Exhaust the stream once.
    query = list(streamer)
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
        data_results.append(isinstance(x, dict) and 'X' in x)
        if (i + 1) >= count_max:
            break
    assert (len(data_results) == count_max and all(data_results))


@pytest.mark.parametrize('max_iter', [3, 10])
def test_streamer_cycle_maxiter(max_iter):

    s = pescador.Streamer(T.finite_generator, 6)

    r1 = list(s.cycle(max_iter=max_iter))
    assert len(r1) == max_iter

    r2 = list(s(max_iter=max_iter, cycle=True))
    assert len(r2) == max_iter


def test_streamer_bad_function():

    def __fail():
        return 6

    with pytest.raises(pescador.core.PescadorError):
        pescador.Streamer(__fail)


def test_streamer_copy():
    stream_len = 10
    streamer = pescador.core.Streamer(T.finite_generator, stream_len)

    s_copy = copy.copy(streamer)
    assert streamer is not s_copy
    assert streamer.streamer is s_copy.streamer
    assert streamer.args is s_copy.args
    assert streamer.kwargs is s_copy.kwargs
    assert streamer.active_count_ == s_copy.active_count_
    assert streamer.stream_ is s_copy.stream_


def test_streamer_deepcopy():
    stream_list = list(range(100))
    stream_len = (10,)
    kwargs = dict(a=10, b=20, c=30)

    # As stream_list is not callable, the streamer won't actually pass the
    # args or kwargs to it, but we can still check if they got copied!
    streamer = pescador.core.Streamer(stream_list,
                                      *stream_len, **kwargs)

    s_copy = copy.deepcopy(streamer)
    assert streamer is not s_copy
    assert streamer.streamer is not s_copy.streamer
    # args is a tuple and is immutable, so it won't actually get deepcopied.
    assert streamer.args is s_copy.args
    # But the kwargs dict will get a correct deepcopy.
    assert streamer.kwargs is not s_copy.kwargs

    assert streamer.streamer == s_copy.streamer
    assert streamer.args == s_copy.args
    assert streamer.kwargs == s_copy.kwargs
    assert streamer.active_count_ == s_copy.active_count_
    assert streamer.stream_ == s_copy.stream_


def test_streamer_context_copy():
    """Check that the streamer produced by __enter__/activate
    is a *different* streamer than the original.

    Note: Do not use the streamer in this way in your code! You
    can't actually extract samples from the streamer using the context
    manager externally.
    """
    stream_len = 10
    streamer = pescador.core.Streamer(T.finite_generator, stream_len)
    assert streamer.stream_ is None
    assert streamer.active == 0

    with streamer as active_stream:
        # the original streamer should be makred active now
        assert streamer.active == 1
        # The reference shouldn't be active
        assert active_stream.active == 0

        assert isinstance(active_stream, pescador.core.Streamer)
        # Check that the objects are not the same
        assert active_stream is not streamer

        assert streamer.stream_ is None
        # The active stream should have been activated.
        assert active_stream.stream_ is not None
        assert active_stream.streamer == streamer.streamer
        assert active_stream.args == streamer.args
        assert active_stream.kwargs == streamer.kwargs

        assert active_stream.is_activated_copy is True

        # Now, we should be able to iterate on active_stream without it
        # causing another copy.
        with active_stream as test_stream:
            assert active_stream is test_stream
            assert streamer.active == 1

        # Exhaust the stream once.
        query = list(active_stream)
        assert stream_len == len(query)

    assert streamer.active == 0


def test_streamer_context_multiple_copies():
    """Check that a streamer produced by __enter__/activate
    multiple times yields streamers *different* than the original.
    """
    stream_len = 10
    streamer = pescador.core.Streamer(T.finite_generator, stream_len)
    assert streamer.stream_ is None
    assert streamer.active == 0

    # Active the streamer multiple times with iterate
    gen1 = streamer.iterate(5)
    gen2 = streamer.iterate(7)
    assert id(gen1) != id(gen2)

    # No streamers should be active until we actually start the generators
    assert streamer.active == 0

    # grab one sample each to make sure we've actually started the generator
    _ = next(gen1)
    _ = next(gen2)
    assert streamer.active == 2

    # the first one should die after four more samples
    result1 = list(gen1)
    assert len(result1) == 4
    assert streamer.active == 1

    # The second should die after 6
    result2 = list(gen2)
    assert len(result2) == 6
    assert streamer.active == 0


def test_decorator():

    @pescador.streamable
    def my_generator(n):
        yield from range(n)

    s = my_generator(5)
    assert isinstance(s, pescador.Streamer)
    assert list(s) == [0, 1, 2, 3, 4]

