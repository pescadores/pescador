import six
import pescador
import test_utils as T

import warnings
warnings.simplefilter('always')


def test_zmq():

    def __test(copy, timeout):
        stream = pescador.Streamer(T.finite_generator, 200, size=3, lag=0.001)

        reference = list(stream.generate())

        zmq_stream = pescador.ZMQStreamer(stream, copy=copy, timeout=timeout)

        for _ in range(3):
            query = list(zmq_stream.generate())
            eq_(len(reference), len(query))
            for b1, b2 in zip(reference, query):
                T.__eq_batch(b1, b2)

    for copy in [False, True]:
        for timeout in [None, 0.5, 2, 5]:
            yield __test, copy, timeout


def test_zmq_align():

    stream = pescador.Streamer(T.finite_generator, 200, size=3, lag=0.001)

    reference = list(stream.generate())
    warnings.resetwarnings()

    zmq_stream = pescador.ZMQStreamer(stream)
    with warnings.catch_warnings(record=True) as out:
        query = list(zmq_stream.generate())
        eq_(len(reference), len(query))

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
