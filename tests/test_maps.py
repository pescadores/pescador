import pytest

import numpy as np

import pescador.maps
import test_utils as T


def test___stack_data():
    n_items = 10
    data = [{"X": np.array([n])} for n in range(n_items)]
    expected = {"X": np.arange(n_items).reshape(-1, 1)}
    output = pescador.maps.__stack_data(data)
    T.__eq_batch(expected, output)


def test_buffer_stream():
    inputs = [{"X": np.array([n])} for n in range(10)]
    expected = [{"X": np.array([0, 1, 2, 3]).reshape(-1, 1)},
                {"X": np.array([4, 5, 6, 7]).reshape(-1, 1)},
                {"X": np.array([8, 9]).reshape(-1, 1)}]

    stream = pescador.maps.buffer_stream(inputs, buffer_size=4)
    outputs = list(stream)
    assert len(outputs) == (len(expected) - 1)
    for exp, obs in zip(expected, outputs):
        T.__eq_batch(exp, obs)

    stream = pescador.maps.buffer_stream(inputs, buffer_size=4, partial=True)
    outputs = list(stream)
    assert len(outputs) == len(expected)
    for exp, obs in zip(expected, outputs):
        T.__eq_batch(exp, obs)


def test_keras_tuples():
    data = [{"foo": np.array([n]), "bar": np.array([n / 2.]),
             "whiz": np.array([2 * n]), "bang": np.array([n ** 2])}
            for n in range(10)]

    stream = pescador.maps.keras_tuples(data, inputs="foo", outputs="bar")
    for n, (x, y) in enumerate(stream):
        assert n == x == y * 2

    stream = pescador.maps.keras_tuples(data, outputs="whiz")
    for n, (x, y) in enumerate(stream):
        assert n == y / 2.
        assert x is None

    stream = pescador.maps.keras_tuples(data, inputs="bang")
    for n, (x, y) in enumerate(stream):
        assert n == x ** 0.5
        assert y is None

    stream = pescador.maps.keras_tuples(data, inputs=["foo", "bang"],
                                        outputs=["bar", "whiz"])
    for n, (x, y) in enumerate(stream):
        assert len(x) == len(y) == 2
