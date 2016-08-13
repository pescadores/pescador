from nose.tools import raises, eq_

import pescador
import test_utils as T


def test_mux_single():

    reference = list(T.finite_generator(50))
    stream = pescador.Streamer(reference)

    estimate = pescador.mux([stream], None, 1, with_replacement=False)
    eq_(list(reference), list(estimate))


@raises(RuntimeError)
def test_mux_empty():

    list(pescador.mux([], None, 1))


def test_mux_weighted():

    def __test(weight):
        reference = list(T.finite_generator(50))
        noise = list(T.finite_generator(50, size=1))
        stream = pescador.Streamer(reference)
        stream2 = pescador.Streamer(noise)
        estimate = pescador.mux([stream, stream2], None, 2,
                                pool_weights=[1.0, weight],
                                with_replacement=False)
        eq_(list(reference), list(estimate))

    yield __test, 0.0
    yield raises(AssertionError)(__test), 0.5


def test_mux_rare():

    def __test(weight):
        reference = list(T.finite_generator(50))
        noise = list(T.finite_generator(50, size=1))
        stream = pescador.Streamer(reference)
        stream2 = pescador.Streamer(noise)
        estimate = pescador.mux([stream, stream2], None, 2,
                                pool_weights=weight,
                                with_replacement=False)
        eq_(list(reference) + list(noise), list(estimate))

    # This should give us all the reference before all the noise
    yield __test, [1e10, 1e-10]


def test_empty_seeds():

    def __empty():
        if False:
            yield 1

    reference = pescador.Streamer(T.finite_generator, 10)
    empty = pescador.Streamer(__empty)

    estimate = pescador.mux([reference, empty], 10, 2, lam=None,
                            with_replacement=False,
                            pool_weights=[1e-10, 1e10])

    estimate = list(estimate)

    ref = list(reference.generate())
    eq_(len(ref), len(estimate))
    for b1, b2 in zip(ref, estimate):
        T.__eq_batch(b1, b2)


def test_mux_replacement():

    def __test(n_streams, n_samples, k, lam):

        seeds = [pescador.Streamer(T.infinite_generator)
                 for _ in range(n_streams)]

        mux = pescador.mux(seeds, n_samples, k, lam=lam)

        estimate = list(mux)

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

        mux = pescador.mux(seeds, n_samples, k, lam=lam,
                           with_replacement=False,
                           revive=True)

        estimate = list(mux)

        # Make sure we get the right number of samples
        # This is highly improbable when revive=False
        eq_(len(estimate), n_samples)

    for n_streams in [1, 2, 4]:
        for n_samples in [512]:
            for k in [1, 2, 4]:
                for lam in [1.0, 2.0, 4.0]:
                    yield __test, n_streams, n_samples, k, lam
