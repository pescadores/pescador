import pytest
import numpy as np
import six
import pescador
import test_utils as T

import warnings
warnings.simplefilter('always')


@pytest.mark.parametrize('copy', [False, True])
@pytest.mark.parametrize('timeout', [None, 0.5, 2, 5])
def test_zmq(copy, timeout):
    stream = pescador.Streamer(T.finite_generator, 200, size=3, lag=0.001)
    reference = list(stream.generate())

    zmq_stream = pescador.ZMQStreamer(stream, copy=copy, timeout=timeout)

    for _ in range(3):
        query = list(zmq_stream.generate())
        assert len(reference) == len(query)
        for b1, b2 in zip(reference, query):
            T.__eq_batch(b1, b2)


@pytest.mark.parametrize('items',
                         [['X'], ['Y'], ['X', 'Y'], ['Y', 'X'],
                          pytest.mark.xfail([],
                                            raises=pescador.PescadorError)])
def test_zmq_tuple(items):

    stream = pescador.Streamer(T.md_generator, 2, 50, items=items)
    reference = list(stream.generate())

    zmq_stream = pescador.ZMQStreamer(stream, timeout=5)

    estimate = list(zmq_stream.tuples(*items))

    assert len(reference) == len(estimate)
    for r, e in zip(reference, estimate):
        assert isinstance(e, tuple)
        for item, value in zip(items, e):
            assert np.allclose(r[item], value)


def test_zmq_align():

    stream = pescador.Streamer(T.finite_generator, 200, size=3, lag=0.001)

    reference = list(stream.generate())
    warnings.resetwarnings()

    zmq_stream = pescador.ZMQStreamer(stream)
    with warnings.catch_warnings(record=True) as out:
        query = list(zmq_stream.generate())
        assert len(reference) == len(query)

        if six.PY2:
            assert len(out) > 0
            assert out[0].category is RuntimeWarning
            assert 'align' in str(out[0].message).lower()

        for b1, b2 in zip(reference, query):
            T.__eq_batch(b1, b2)
            if six.PY2:
                continue
            for key in b2:
                assert b2[key].flags['ALIGNED']
