import pytest

import numpy as np

import pescador.maps
import test_utils as T


def test___stack_data():
    n_items = 10
    data = [{"X": np.array([n])} for n in range(n_items)]
    expected = {"X": np.arange(n_items).reshape(-1, 1)}
    output = pescador.maps.__stack_data(data, None)
    T._eq_batch(expected, output)


def test___stack_data_axis():
    # This tests data stacking along a chosen axis
    # test data are [[n]]
    n_items = 10
    data = [{"X": np.array([[n]])} for n in range(n_items)]
    expected = {'X': np.arange(n_items).reshape(-1, 1)}
    output = pescador.maps.__stack_data(data, 0)
    T._eq_batch(expected, output)


@pytest.mark.parametrize('axis', [None, 0])
def test_buffer_stream(axis):
    if axis is None:
        inputs = [{"X": np.array([n])} for n in range(10)]
    else:
        inputs = [{"X": np.array([[n]])} for n in range(10)]

    expected = [{"X": np.array([0, 1, 2, 3]).reshape(-1, 1)},
                {"X": np.array([4, 5, 6, 7]).reshape(-1, 1)},
                {"X": np.array([8, 9]).reshape(-1, 1)}]

    stream = pescador.maps.buffer_stream(inputs, buffer_size=4, axis=axis)
    outputs = list(stream)
    assert len(outputs) == (len(expected) - 1)
    for exp, obs in zip(expected, outputs):
        T._eq_batch(exp, obs)

    stream = pescador.maps.buffer_stream(inputs, buffer_size=4,
                                         partial=True, axis=axis)
    outputs = list(stream)
    assert len(outputs) == len(expected)
    for exp, obs in zip(expected, outputs):
        T._eq_batch(exp, obs)

    with pytest.raises(pescador.maps.DataError):
        for not_data in pescador.maps.buffer_stream([1, 2, 3, 4], 2,
                                                    axis=axis):
            pass


@pytest.fixture
def sample_data():
    return [{"foo": np.array([n]), "bar": np.array([n / 2.]),
             "whiz": np.array([2 * n]), "bang": np.array([n ** 2])}
            for n in range(10)]


def test_tuples(sample_data):

    stream = pescador.maps.tuples(sample_data, "foo", "bar")
    for n, (x, y) in enumerate(stream):
        assert n == x == y * 2

    stream = pescador.maps.tuples(sample_data, "whiz")
    for n, (x,) in enumerate(stream):
        assert n == x / 2.

    with pytest.raises(pescador.maps.PescadorError):
        for x in pescador.maps.tuples(sample_data):
            pass

    with pytest.raises(pescador.maps.DataError):
        for x in pescador.maps.tuples([1, 2, 3], 'baz'):
            pass

    with pytest.raises(KeyError):
        for x in pescador.maps.tuples(sample_data, 'apple'):
            pass


def test_keras_tuples(sample_data):

    stream = pescador.maps.keras_tuples(sample_data, inputs="foo",
                                        outputs="bar")
    for n, (x, y) in enumerate(stream):
        assert n == x == y * 2

    stream = pescador.maps.keras_tuples(sample_data, outputs="whiz")
    for n, (x, y) in enumerate(stream):
        assert n == y / 2.
        assert x is None

    stream = pescador.maps.keras_tuples(sample_data, inputs="bang")
    for n, (x, y) in enumerate(stream):
        assert n == x ** 0.5
        assert y is None

    stream = pescador.maps.keras_tuples(sample_data, inputs=["bang"])
    for n, (x, y) in enumerate(stream):
        assert n == x[0] ** 0.5
        assert y is None

    stream = pescador.maps.keras_tuples(sample_data, inputs=["foo", "bang"],
                                        outputs=["bar", "whiz"])
    for n, (x, y) in enumerate(stream):
        assert len(x) == len(y) == 2

    with pytest.raises(pescador.maps.PescadorError):
        for x in pescador.maps.keras_tuples(sample_data):
            pass

    with pytest.raises(pescador.maps.DataError):
        for x in pescador.maps.keras_tuples([1, 2, 3], 'baz'):
            pass

    with pytest.raises(KeyError):
        for x in pescador.maps.keras_tuples(sample_data, 'apple'):
            pass


@pytest.mark.parametrize(
    'n_cache',
    [2, 4, 8, 64,
     pytest.param(-1, marks=pytest.mark.xfail(raises=pescador.PescadorError))]
)
@pytest.mark.parametrize(
    'prob',
    [0.1, 0.5, 1,
     pytest.param(-1, marks=pytest.mark.xfail(raises=pescador.PescadorError)),
     pytest.param(0, marks=pytest.mark.xfail(raises=pescador.PescadorError)),
     pytest.param(1.5, marks=pytest.mark.xfail(raises=pescador.PescadorError))]
)
def test_cache(n_cache, prob):
    data = list(range(32))

    cache = pescador.maps.cache(iter(range(32)), n_cache, prob, random_state=0)

    output = list(cache)

    if n_cache >= len(data) or prob == 1.0:
        T._eq_lists(data, output), (data, output)
    else:
        assert len(output) >= len(data)
