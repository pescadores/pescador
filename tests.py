import pescador
import numpy as np


def test_buffer_stream():
    stream = pescador.buffer_stream(range(3), 2, 1)
    for exp, act in zip([[0, 1]], stream):
        assert exp == act

    stream = pescador.buffer_stream(range(3), 2, 2)
    for exp, act in zip([[0, 1], [2]], stream):
        assert exp == act


def test_mux_without_replacement():
    N = 50
    iter_pool = [pescador.Streamer(iter, [(n,)]) for n in range(N)]
    mux = pescador.mux(iter_pool, None, 10, with_replacement=False)
    result = [value for value in mux]
    result.sort()
    assert result == [(n,) for n in range(N)]


def test_mux_prune_empty_seeds():
    n_samples = 10
    iters = [[], ['a']]
    iter_pool = [pescador.Streamer(iter, x) for x in iters]
    mux = pescador.mux(iter_pool, n_samples=10, k=10, with_replacement=True)
    result = [value for value in mux]
    expected = ['a']*n_samples
    assert result == expected, "%s, %s" % (result, expected)


def test_generate_new_stream():
    iters = [[1], [2]]
    iter_pool = [pescador.Streamer(iter, x) for x in iters]
    weights = [0.2, 0.8]
    distribution = np.array([0.4, 0.6])
    stream, weight = pescador._generate_new_seed(
        0, iter_pool, weights, distribution, with_replacement=True)
    assert stream.next() == 1
    assert weight == 0.2
    assert distribution.tolist() == [0.4, 0.6]

    stream, weight = pescador._generate_new_seed(
        1, iter_pool, weights, distribution, with_replacement=False)
    assert stream.next() == 2
    assert weight == 0.8
    assert distribution.tolist() == [1.0, 0.0]
