from nose.tools import raises, eq_
import pescador

import test_utils as T


def test_buffer_streamer():

    def __serialize_batches(batches):

        for batch in batches:
            for item in batch['X']:
                yield item

    def __test(dimension, n_batch, n_buf):
        reference = T.md_generator(dimension, 50, size=n_batch)

        reference = list(__serialize_batches(reference))

        gen_stream = pescador.Streamer(T.md_generator, dimension, 50,
                                       size=n_batch)
        estimate = pescador.BufferedStreamer(gen_stream, n_buf)

        estimate = list(__serialize_batches(estimate.generate()))

        T.__eq_lists(reference, estimate)

    for dimension in [1, 2, 3]:
        for batch_size in [1, 2, 5, 17]:
            for buf_size in [1, 2, 5, 17, 100]:
                yield __test, dimension, batch_size, buf_size


def test_batch_length():
    def __test(generator, n):
        for batch in generator:
            T.eq_(pescador.util.batch_length(batch), n)

    for n1 in [5, 10, 15]:
        for n2 in [5, 10, 15]:
            if n1 != n2:
                test = raises(RuntimeError)(__test)
            else:
                test = __test
            yield test, T.__zip_generator(3, n1, n2), n1


def test_buffer_batch():

    def __serialize_batches(batches):

        for batch in batches:
            for item in batch['X']:
                yield item

    def __test(dimension, n_batch, n_buf):
        reference = T.md_generator(dimension, 50, size=n_batch)

        reference = list(__serialize_batches(reference))

        estimate = pescador.buffer_batch(T.md_generator(dimension,
                                                      50,
                                                      size=n_batch),
                                         n_buf)

        estimate = list(__serialize_batches(estimate))

        T.__eq_lists(reference, estimate)

    for dimension in [1, 2, 3]:
        for batch_size in [1, 2, 5, 17]:
            for buf_size in [1, 2, 5, 17, 100]:
                yield __test, dimension, batch_size, buf_size
