#!/usr/bin/env python
# TODO: Remove these tests with the `buffered.py` submodule at 2.0 release.

import pytest
import numpy as np

import pescador
import pescador.buffered
import test_utils as T


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('batch_size', [1, 2, 5, 17])
@pytest.mark.parametrize('buf_size', [1, 2, 5, 17, 100])
def test_BufferedStreamer(dimension, batch_size, buf_size):

    key = 'X'

    def __unpack_stream(stream):
        for data in stream:
            for item in data[key]:
                yield item

    reference = T.md_generator(dimension, 50, size=batch_size)

    reference = [data[key] for data in reference]

    gen_stream = pescador.Streamer(T.md_generator, dimension, 50,
                                   size=batch_size)
    estimate = pescador.BufferedStreamer(gen_stream, buf_size)

    estimate = list(__unpack_stream(estimate))

    T._eq_lists(reference, estimate)

    estimate = pescador.BufferedStreamer(gen_stream, buf_size)

    assert len(list(estimate.iterate(max_iter=2))) <= 2


@pytest.mark.parametrize('items',
                         [['X'], ['Y'], ['X', 'Y'], ['Y', 'X'],
                          pytest.mark.xfail(
                             [], raises=pescador.PescadorError)])
@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('batch_size', [1, 2, 5, 17])
@pytest.mark.parametrize('buf_size', [1, 2, 5, 17, 100])
def test_BufferedStreamer_tuples(dimension, batch_size, buf_size, items):

    gen_stream = pescador.Streamer(T.md_generator, dimension, 50,
                                   size=batch_size, items=items)

    buf = pescador.BufferedStreamer(gen_stream, buf_size)
    estimate = list(buf.tuples(*items))
    reference = list(buf)

    for b, t in zip(reference, estimate):
        assert isinstance(t, tuple)
        assert len(t) == len(items)
        for item, ti in zip(items, t):
            assert np.allclose(b[item], ti)
