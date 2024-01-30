import pytest
import numpy as np
import pescador
import test_utils as T


@pytest.mark.parametrize('copy', [False, True])
@pytest.mark.parametrize('timeout', [None, 0.5, 2, 5])
def test_zmq(copy, timeout):
    stream = pescador.Streamer(T.finite_generator, 200, size=3, lag=0.001)
    reference = list(stream)

    zmq_stream = pescador.ZMQStreamer(stream, copy=copy, timeout=timeout)

    for _ in range(3):
        query = list(zmq_stream)
        assert len(reference) == len(query)
        for b1, b2 in zip(reference, query):
            T._eq_batch(b1, b2)


def test_zmq_align():

    stream = pescador.Streamer(T.finite_generator, 200, size=3, lag=0.001)

    reference = list(stream)

    zmq_stream = pescador.ZMQStreamer(stream)

    query = list(zmq_stream)

    assert len(reference) == len(query)

    for b1, b2 in zip(reference, query):
        T._eq_batch(b1, b2)
        for key in b2:
            assert b2[key].flags['ALIGNED']


def __bad_generator():
    for _ in range(100):
        yield dict(X=list(range(100)))


@pytest.mark.xfail(raises=pescador.PescadorError)
def test_zmq_bad_type():

    stream = pescador.Streamer(__bad_generator)

    zs = pescador.ZMQStreamer(stream)

    for item in zs:
        pass


def test_zmq_early_stop():
    stream = pescador.Streamer(T.finite_generator, 200, size=3, lag=0.001)

    zmq_stream = pescador.ZMQStreamer(stream)

    # Only sample five batches
    assert len([x for x in zip(zmq_stream, range(5))]) == 5


def test_zmq_buffer():
    n_samples = 50
    stream = pescador.Streamer(T.md_generator, dimension=2, n=n_samples,
                               size=64, items=['X', 'Y'])

    buff_size = 10
    buff_stream = pescador.Streamer(pescador.buffer_stream, stream, buff_size)
    zmq_stream = pescador.ZMQStreamer(buff_stream)

    outputs = [x for x in zmq_stream]
    assert len(outputs) == int(n_samples) / buff_size
