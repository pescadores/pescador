import pytest

import numpy as np
import time
import warnings

import pescador.util


def _eq_batch(b1, b2):

    for k in b1:
        assert np.allclose(b1[k], b2[k])


def _eq_lists(b1, b2):

    assert len(b1) == len(b2)

    for i, j in zip(b1, b2):
        assert np.allclose(i, j)


def _eq_list_of_dicts(b1, b2):
    results = []
    results.append(len(b1) == len(b2))

    if results[-1]:
        for i in range(len(b1)):
            for k in b1[i]:
                results.append(np.allclose(b1[i][k], b2[i][k]))

    return np.all(results)


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


def infinite_generator(size=2, offset=0):

    i = 0
    while True:
        yield {'X': np.tile(np.array([[i]]), (size, 1)) + offset}
        i = i + 1


def __zip_generator(n, size1, size2):

    for b1, b2 in zip(finite_generator(n, size=size1),
                      finite_generator(n, size=size2)):
        yield dict(X=b1['X'], Y=b2['X'])


@pytest.mark.parametrize(
    'n1, n2',
    [pytest.param(5, 10, marks=pytest.mark.xfail(raises=pescador.util.PescadorError)),
     pytest.param(5, 15, marks=pytest.mark.xfail(raises=pescador.util.PescadorError)),
     pytest.param(10, 5, marks=pytest.mark.xfail(raises=pescador.util.PescadorError)),
     pytest.param(15, 5, marks=pytest.mark.xfail(raises=pescador.util.PescadorError)),
     (5, 5), (10, 10), (15, 15)]
)
def test_batch_length(n1, n2):
    generator, n = __zip_generator(3, n1, n2), n1

    for batch in generator:
        assert pescador.util.batch_length(batch) == n


def test_warning_deprecated():

    @pescador.util.deprecated('old_version', 'new_version')
    def __dummy():
        return True

    warnings.resetwarnings()
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as out:
        x = __dummy()

        # Make sure we still get the right value
        assert x is True

        # And that the warning triggered
        assert len(out) > 0

        # And that the category is correct
        assert out[0].category is DeprecationWarning

        # And that it says the right thing (roughly)
        assert 'deprecated' in str(out[0].message).lower()


def test_warning_moved():

    @pescador.util.moved('from', 'old_version', 'new_version')
    def __dummy():
        return True

    warnings.resetwarnings()
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as out:
        x = __dummy()

        # Make sure we still get the right value
        assert x is True

        # And that the warning triggered
        assert len(out) > 0

        # And that the category is correct
        assert out[0].category is DeprecationWarning

        # And that it says the right thing (roughly)
        assert 'moved' in str(out[0].message).lower()


def test_warning_rename_kw_pass():
    warnings.resetwarnings()
    warnings.simplefilter('always')

    ov = pescador.util.Deprecated()
    nv = 23

    with warnings.catch_warnings(record=True) as out:
        v = pescador.util.rename_kw('old', ov, 'new', nv, '0', '1')

        assert v == nv

        # Make sure no warning triggered
        assert len(out) == 0


def test_warning_rename_kw_fail():
    warnings.resetwarnings()
    warnings.simplefilter('always')

    ov = 27
    nv = 23

    with warnings.catch_warnings(record=True) as out:
        v = pescador.util.rename_kw('old', ov, 'new', nv, '0', '1')

        assert v == ov

        # Make sure the warning triggered
        assert len(out) > 0

        # And that the category is correct
        assert out[0].category is DeprecationWarning

        # And that it says the right thing (roughly)
        assert 'renamed' in str(out[0].message).lower()
