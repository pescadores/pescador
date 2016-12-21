from nose.tools import raises, eq_

import numpy as np

import pescador
import pescador.mux
import test_utils as T


def test_mux_single():

    reference = list(T.finite_generator(50))
    stream = pescador.Streamer(reference)

    mux = pescador.mux.Mux([stream], 1, with_replacement=False)
    estimate = mux.generate()
    eq_(list(reference), list(estimate))


@raises(pescador.PescadorError)
def test_mux_empty():

    list(pescador.mux.Mux([], 1).generate())


def test_mux_weighted():

    def __test(weight):
        reference = list(T.finite_generator(50))
        noise = list(T.finite_generator(50, size=1))
        stream = pescador.Streamer(reference)
        stream2 = pescador.Streamer(noise)
        mux = pescador.mux.Mux([stream, stream2], 2,
                               pool_weights=[1.0, weight],
                               with_replacement=False)
        estimate = mux.generate()
        eq_(list(reference), list(estimate))

    yield __test, 0.0
    yield raises(AssertionError)(__test), 0.5


def test_mux_rare():

    def __test(weight):
        reference = list(T.finite_generator(50))
        noise = list(T.finite_generator(50, size=1))
        stream = pescador.Streamer(reference)
        stream2 = pescador.Streamer(noise)
        mux = pescador.mux.Mux([stream, stream2], 2,
                               pool_weights=weight,
                               with_replacement=False)
        estimate = mux.generate()
        eq_(list(reference) + list(noise), list(estimate))

    # This should give us all the reference before all the noise
    yield __test, [1e10, 1e-10]


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
    eq_(len(ref), len(estimate))
    for b1, b2 in zip(ref, estimate):
        T.__eq_batch(b1, b2)


def test_mux_replacement():

    def __test(n_streams, n_samples, k, lam):

        seeds = [pescador.Streamer(T.infinite_generator)
                 for _ in range(n_streams)]

        mux = pescador.mux.Mux(seeds, k, lam=lam)

        estimate = list(mux.generate(n_samples))

        # Make sure we get the right number of samples
        eq_(len(estimate), n_samples)

    for n_streams in [1, 2, 4]:
        for n_samples in [10, 20, 80]:
            for k in [1, 2, 4]:
                for lam in [1.0, 2.0, 8.0]:
                        yield __test, n_streams, n_samples, k, lam


def test_mux_revive():

    def __test(n_streams, n_samples, k, lam):

        seeds = [pescador.Streamer(T.finite_generator, 10)
                 for _ in range(n_streams)]

        mux = pescador.mux.Mux(seeds, k, lam=lam,
                               with_replacement=False,
                               revive=True)

        estimate = list(mux.generate(n_samples))

        # Make sure we get the right number of samples
        # This is highly improbable when revive=False
        eq_(len(estimate), n_samples)

    for n_streams in [1, 2, 4]:
        for n_samples in [512]:
            for k in [1, 2, 4]:
                for lam in [1.0, 2.0, 4.0]:
                    yield __test, n_streams, n_samples, k, lam


@raises(pescador.PescadorError)
def test_mux_bad_pool():

    seeds = [pescador.Streamer(T.finite_generator, 10)
             for _ in range(5)]

    # 5 seeds, 10 weights, should trigger an error
    M = pescador.Mux(seeds, None, pool_weights=np.random.randn(10))


@raises(pescador.PescadorError)
def test_mux_bad_weights():

    seeds = [pescador.Streamer(T.finite_generator, 10)
             for _ in range(5)]

    # 5 seeds, all-zeros weight vector should trigger an error
    M = pescador.Mux(seeds, None, pool_weights=np.zeros(5))
