import pytest

import numpy as np
import six
import time

import pescador.util


def __eq_batch(b1, b2):

    for k in six.iterkeys(b1):
        assert np.allclose(b1[k], b2[k])


def __eq_lists(b1, b2):

    assert len(b1) == len(b2)

    for i, j in zip(b1, b2):
        assert np.allclose(i, j)


def finite_generator(n, size=2, lag=None):

    for i in range(n):
        yield {'X': np.tile(np.array([[i]]), (size, 1))}
        if lag is not None:
            time.sleep(lag)


def md_generator(dimension, n, size=2, items='X'):
    """Produce `n` dicts of `dimension`-rank arrays under the names in `items`.
    """
    shape = [size] * dimension

    M = len(items)
    for i in range(n):

        yield {item: i * M * np.ones(shape) + j
               for j, item in enumerate(items)}


def infinite_generator(size=2):

    i = 0
    while True:
        yield {'X': np.tile(np.array([[i]]), (size, 1))}
        i = i + 1


def __zip_generator(n, size1, size2):

    for b1, b2 in zip(finite_generator(n, size=size1),
                      finite_generator(n, size=size2)):
        yield dict(X=b1['X'], Y=b2['X'])


@pytest.mark.parametrize(
    'n1,n2', [pytest.mark.xfail((5, 10), raises=pescador.util.PescadorError),
              pytest.mark.xfail((5, 15), raises=pescador.util.PescadorError),
              pytest.mark.xfail((10, 5), raises=pescador.util.PescadorError),
              pytest.mark.xfail((15, 5), raises=pescador.util.PescadorError),
              (5, 5), (10, 10), (15, 15)])
def test_batch_length(n1, n2):
    generator, n = __zip_generator(3, n1, n2), n1

    for batch in generator:
        assert pescador.util.batch_length(batch) == n
