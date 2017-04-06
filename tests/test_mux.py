import pytest

import numpy as np

import pescador
import pescador.mux
import test_utils as T


def test_mux_single():

    reference = list(T.finite_generator(50))
    stream = pescador.Streamer(reference)

    mux = pescador.mux.Mux([stream], 1, with_replacement=False)
    estimate = mux.generate()
    assert list(reference) == list(estimate)


@pytest.mark.parametrize('items',
                         [['X'], ['Y'], ['X', 'Y'], ['Y', 'X'],
                          pytest.mark.xfail([],
                                            raises=pescador.PescadorError)])
def test_mux_single_tuple(items):

    stream = pescador.Streamer(T.md_generator, 2, 50, items=items)
    reference = list(stream.generate())

    mux = pescador.mux.Mux([stream], 1, with_replacement=False)
    estimate = list(mux.tuples(*items))

    assert len(reference) == len(estimate)
    for r, e in zip(reference, estimate):
        assert isinstance(e, tuple)
        for item, value in zip(items, e):
            assert np.allclose(r[item], value)


def test_mux_empty():
    with pytest.raises(pescador.PescadorError):
        list(pescador.mux.Mux([], 1).generate())


@pytest.mark.parametrize('weight', [0.0, 0.5])
def test_mux_weighted(weight):
    reference = list(T.finite_generator(50))
    noise = list(T.finite_generator(50, size=1))
    stream = pescador.Streamer(reference)
    stream2 = pescador.Streamer(noise)
    mux = pescador.mux.Mux([stream, stream2], 2,
                           pool_weights=[1.0, weight],
                           with_replacement=False)
    estimate = mux.generate()
    if weight == 0.0:
        assert list(reference) == list(estimate)
    else:
        assert list(reference) != list(estimate)


# This should give us all the reference before all the noise
@pytest.mark.parametrize('weight', ([1e10, 1e-10],))
def test_mux_rare(weight):
    reference = list(T.finite_generator(50))
    noise = list(T.finite_generator(50, size=1))
    stream = pescador.Streamer(reference)
    stream2 = pescador.Streamer(noise)
    mux = pescador.mux.Mux([stream, stream2], 2,
                           pool_weights=weight,
                           with_replacement=False)
    estimate = mux.generate()
    assert (list(reference) + list(noise)) == list(estimate)


def test_empty_seeds():

    def __empty():
        if False:
            yield 1

    reference = pescador.Streamer(T.finite_generator, 10)
    empty = pescador.Streamer(__empty)

    mux = pescador.mux.Mux([reference, empty], 2, lam=None,
                           with_replacement=False,
                           pool_weights=[1e-10, 1e10])
    estimate = mux.generate(10)
    estimate = list(estimate)

    ref = list(reference.generate())
    assert len(ref) == len(estimate)
    for b1, b2 in zip(ref, estimate):
        T.__eq_batch(b1, b2)


@pytest.mark.parametrize('n_streams', [1, 2, 4])
@pytest.mark.parametrize('n_samples', [10, 20, 80])
@pytest.mark.parametrize('k', [1, 2, 4])
@pytest.mark.parametrize('lam', [1.0, 2.0, 8.0])
@pytest.mark.parametrize('random_state',
                         [None,
                          1000,
                          np.random.RandomState(seed=1000),
                          pytest.mark.xfail('foo',
                                            raises=pescador.PescadorError)])
def test_mux_replacement(n_streams, n_samples, k, lam, random_state):
    seeds = [pescador.Streamer(T.infinite_generator)
             for _ in range(n_streams)]

    mux = pescador.mux.Mux(seeds, k, lam=lam, random_state=random_state)

    estimate = list(mux.generate(n_samples))

    # Make sure we get the right number of samples
    assert len(estimate) == n_samples


@pytest.mark.parametrize('n_streams', [1, 2, 4])
@pytest.mark.parametrize('n_samples', [512])
@pytest.mark.parametrize('k', [1, 2, 4])
@pytest.mark.parametrize('lam', [1.0, 2.0, 4.0])
def test_mux_revive(n_streams, n_samples, k, lam):
    seeds = [pescador.Streamer(T.finite_generator, 10)
             for _ in range(n_streams)]

    mux = pescador.mux.Mux(seeds, k, lam=lam,
                           with_replacement=False,
                           revive=True)

    estimate = list(mux.generate(n_samples))

    # Make sure we get the right number of samples
    # This is highly improbable when revive=False
    assert len(estimate) == n_samples


def test_mux_bad_pool():
    with pytest.raises(pescador.PescadorError):
        seeds = [pescador.Streamer(T.finite_generator, 10)
                 for _ in range(5)]

        # 5 seeds, 10 weights, should trigger an error
        pescador.Mux(seeds, None, pool_weights=np.random.randn(10))


def test_mux_bad_weights():
    with pytest.raises(pescador.PescadorError):
        seeds = [pescador.Streamer(T.finite_generator, 10)
                 for _ in range(5)]

        # 5 seeds, all-zeros weight vector should trigger an error
        pescador.Mux(seeds, None, pool_weights=np.zeros(5))
