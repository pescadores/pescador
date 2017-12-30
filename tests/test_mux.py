# This makes '/' do in python2 what you expect in python3.
from __future__ import division

import pytest


import collections
import functools
import itertools
import numpy as np
import scipy.stats

import pescador
import pescador.mux
import test_utils as T


def _cycle(values):
    while True:
        for v in values:
            yield v


def _choice(vals, seed=11111):
    rng = np.random.RandomState(seed=seed)
    n = len(vals)
    while True:
        yield vals[rng.randint(0, n)]


@pytest.mark.parametrize('mux_class', [
    functools.partial(pescador.mux.Mux, k=1, with_replacement=False),
    functools.partial(pescador.mux.PoissonMux, k_active=1, rate=None,
                      mode="exhaustive"),
    pescador.mux.RoundRobinMux,
    pescador.mux.ChainMux,
],
    ids=["DeprecatedMux",
         "PoissonMux-exhaustive",
         "RoundRobin",
         "ChainMux",
         ])
def test_mux_single_finite(mux_class):
    "Test a single finite streamer for each mux with an exhaustive setting."

    reference = list(T.finite_generator(50))
    stream = pescador.Streamer(reference)

    mux = mux_class([stream])
    estimate = list(mux)
    assert len(reference) == len(estimate)
    for i, d in enumerate(estimate):
        assert d.items() == estimate[i].items()


@pytest.mark.parametrize('mux_class', [
    functools.partial(pescador.mux.Mux, k=1, with_replacement=True),
    functools.partial(pescador.mux.PoissonMux, k_active=1, rate=None,
                      mode="with_replacement"),
    functools.partial(pescador.mux.PoissonMux, k_active=1, rate=None,
                      mode="single_active"),
    pescador.mux.ShuffledMux,
    functools.partial(pescador.mux.RoundRobinMux, mode="cycle"),
    functools.partial(pescador.mux.ChainMux, mode="cycle"),
],
    ids=["DeprecatedMux",
         "PoissonMux-with_replacement",
         "PoissonMux-single_active",
         "ShuffledMux",
         "RoundRobinMux",
         "ChainMux-cycle"
         ])
def test_mux_single_infinite(mux_class):
    """Test a single finite streamer for each mux class which can revive it's
    streamers.
    """
    reference = list(T.finite_generator(50))
    stream = pescador.Streamer(reference)

    mux = mux_class([stream])
    estimate = list(mux.iterate(max_iter=100))
    assert (reference + reference) == estimate


@pytest.mark.parametrize('mux_class', [
    functools.partial(pescador.mux.Mux, k=1),
    functools.partial(pescador.mux.PoissonMux, k_active=1, rate=256),
    pescador.mux.ShuffledMux,
    pescador.mux.RoundRobinMux,
    pytest.mark.xfail(pescador.mux.ChainMux,
                      reason="ChainMux can accept an empty iterable or "
                             "generator, and will simply return empty.",
                      strict=True),
],
    ids=["DeprecatedMux",
         "PoissonMux-exhaustive",
         "ShuffledMux",
         "RoundRobinMux",
         "ChainMux"
         ])
def test_mux_empty(mux_class):
    "Make sure an empty list of streamers raises an error."
    with pytest.raises(pescador.PescadorError):
        list(mux_class([]))


@pytest.mark.parametrize('mux_class', [
    functools.partial(pescador.mux.Mux, k=None),
    functools.partial(pescador.mux.PoissonMux, k_active=None, rate=256),
    pescador.mux.ShuffledMux,
],
    ids=["DeprecatedMux",
         "PoissonMux",
         "ShuffledMux",
         ])
def test_mux_bad_streamers(mux_class):
    with pytest.raises(pescador.PescadorError):
        steamers = [pescador.Streamer(T.finite_generator, 10)
                    for _ in range(5)]

        # 5 steamers, 10 weights, should trigger an error
        mux_class(steamers, weights=np.random.randn(10))


@pytest.mark.parametrize('mux_class', [
    functools.partial(pescador.mux.Mux, k=None),
    functools.partial(pescador.mux.PoissonMux, k_active=None, rate=256),
    pescador.mux.ShuffledMux,
],
    ids=["DeprecatedMux",
         "PoissonMux",
         "ShuffledMux",
         ])
def test_mux_bad_weights(mux_class):
    with pytest.raises(pescador.PescadorError):
        streamers = [pescador.Streamer(T.finite_generator, 10)
                     for _ in range(5)]

        # 5 streamers, all-zeros weight vector should trigger an error
        mux_class(streamers, weights=np.zeros(5))


class TestPoissonMux:
    @pytest.mark.parametrize(
        'mode', ['with_replacement', 'single_active', 'exhaustive',
                 pytest.mark.xfail('foo', raises=pescador.PescadorError),
                 pytest.mark.xfail(None, raises=pescador.PescadorError)])
    @pytest.mark.parametrize('n_samples', [10])
    def test_valid_modes(self, mode, n_samples):
        """Simply tests the modes to make sure they work."""
        streamers = [pescador.Streamer(T.infinite_generator)
                     for _ in range(4)]

        mux = pescador.mux.PoissonMux(streamers, 1, rate=10, mode=mode)
        output = list(mux.iterate(n_samples))
        assert len(output) == n_samples


@pytest.mark.parametrize('mux_class', [
    functools.partial(pescador.mux.Mux, with_replacement=True),
    functools.partial(pescador.mux.PoissonMux, mode='with_replacement')],
    ids=["DeprecatedMux",
         "PoissonMux"])
class TestPoissonMux_WithReplacement:
    @pytest.mark.parametrize('n_streams', [1, 2, 4])
    @pytest.mark.parametrize('n_samples', [10, 20, 80])
    @pytest.mark.parametrize('k_active', [1, 2, 4])
    @pytest.mark.parametrize('rate', [1.0, 2.0, 8.0])
    @pytest.mark.parametrize('random_state',
                             [None,
                              1000,
                              np.random.RandomState(seed=1000),
                              pytest.mark.xfail(
                                  'foo', raises=pescador.PescadorError,
                                  strict=True),
                              ])
    def test_mux_replacement(self, mux_class, n_streams, n_samples, k_active,
                             rate, random_state):
        streamers = [pescador.Streamer(T.infinite_generator)
                     for _ in range(n_streams)]

        mux = mux_class(streamers, k_active, rate=rate,
                        random_state=random_state)
        estimate = list(mux.iterate(n_samples))

        # Make sure we get the right number of samples
        assert len(estimate) == n_samples

    @pytest.mark.parametrize('n_samples', [10, 20, 80])
    @pytest.mark.parametrize('rate', [1.0, 2.0, 8.0])
    @pytest.mark.parametrize('random_state', [100])
    def test_mux_k_greater_n(self, mux_class, n_samples, rate, random_state):
        """Test that replacement works correctly. See #112:
        https://github.com/pescadores/pescador/issues/112

        When streamers are activated, they should make copies of their
        underlying streamers, and this should work. Before the bug
        was fixed, this would fail. Note; this doesn't test underlying
        state at all, however.
        """
        a = pescador.Streamer('a')
        b = pescador.Streamer('b')

        mux = mux_class([a, b], 6, rate, random_state=random_state)
        result = list(mux.iterate(n_samples))
        assert len(result) == n_samples


class TestPoissonMux_Exhaustive:
    @pytest.mark.parametrize('mux_class', [
        functools.partial(pescador.mux.Mux, with_replacement=False),
        functools.partial(pescador.mux.PoissonMux, mode="exhaustive")],
        ids=["DeprecatedMux",
             "PoissonMux-exhaustive"])
    @pytest.mark.parametrize('weight', [0.0, 0.5])
    def test_mux_weighted(self, weight, mux_class):
        reference = list(T.finite_generator(50))
        noise = list(T.finite_generator(50, size=1))
        stream = pescador.Streamer(reference)
        stream2 = pescador.Streamer(noise)
        mux = mux_class([stream, stream2], 2, rate=256,
                        weights=[1.0, weight])
        estimate = list(mux)
        if weight == 0.0:
            assert reference == estimate
        else:
            assert reference != estimate

    @pytest.mark.parametrize('mux_class', [
        functools.partial(pescador.mux.Mux, with_replacement=False),
        functools.partial(pescador.mux.PoissonMux, mode="exhaustive")],
        ids=["DeprecatedMux",
             "PoissonMux-exhaustive"])
    @pytest.mark.parametrize('weight', ([1e10, 1e-10],))
    def test_mux_rare(self, weight, mux_class):
        "This should give us all the reference before all the noise"
        reference = list(T.finite_generator(50))
        noise = list(T.finite_generator(50, size=1))
        stream = pescador.Streamer(reference)
        stream2 = pescador.Streamer(noise)
        mux = mux_class([stream, stream2], 2, rate=256,
                        weights=weight)
        estimate = list(mux)
        assert (reference + noise) == estimate

    @pytest.mark.parametrize('mux_class', [
        functools.partial(pescador.mux.Mux, k=2, with_replacement=False,
                          rate=None),
        functools.partial(pescador.mux.PoissonMux,
                          k_active=2, mode="exhaustive", rate=None)],
        ids=["DeprecatedMux",
             "PoissonMux-exhaustive"])
    def test_weighted_empty_streams(self, mux_class):

        def __empty():
            if False:
                yield 1

        reference = pescador.Streamer(T.finite_generator, 10)
        empty = pescador.Streamer(__empty)

        mux = mux_class([reference, empty],
                        weights=[1e-10, 1e10])
        estimate = list(mux.iterate(10))

        ref = list(reference)
        assert len(ref) == len(estimate)
        for b1, b2 in zip(ref, estimate):
            T._eq_batch(b1, b2)

    @pytest.mark.parametrize('mux_class', [
        functools.partial(pescador.mux.Mux,
                          with_replacement=False, revive=False),
        functools.partial(pescador.mux.PoissonMux, mode="exhaustive")],
        ids=["DeprecatedMux",
             "PoissonMux"])
    def test_critical_mux(self, mux_class):
        """This test checks the following:

        When `max_iter` is specified, the `Mux` should return / complete
        when the input generators are complete, and should not cycle
        infinitely.

        # Check on Issue #80
        https://github.com/pescadores/pescador/issues/80
        """
        chars = 'abcde'
        n_reps = 7
        streamers = [pescador.Streamer(x * n_reps) for x in chars]
        mux = mux_class(streamers, len(chars), rate=None,
                        prune_empty_streams=False, random_state=135)
        samples = list(mux.iterate(max_iter=1000))
        sample_counts = collections.Counter(samples)
        assert len(sample_counts) == len(chars)
        for k, v in sample_counts.items():
            assert v == n_reps
        assert len(samples) == len(chars) * n_reps

    @pytest.mark.parametrize('mux_class', [
        functools.partial(pescador.mux.Mux,
                          with_replacement=False, revive=False),
        functools.partial(pescador.mux.PoissonMux, mode="exhaustive")],
        ids=["DeprecatedMux",
             "PoissonMux"])
    def test_sampled_mux_of_muxes(self, mux_class):
        # Build some sample streams
        ab = pescador.Streamer(_cycle, 'ab')
        cd = pescador.Streamer(_cycle, 'cd')
        ef = pescador.Streamer(_cycle, 'ef')
        mux1 = mux_class([ab, cd, ef], 3, rate=None)

        # And inspect the first mux
        samples1 = list(mux1(max_iter=6 * 10))
        count1 = collections.Counter(samples1)
        assert set(count1.keys()) == set('abcdef')

        # Build another set of streams
        gh = pescador.Streamer(_cycle, 'gh')
        ij = pescador.Streamer(_cycle, 'ij')
        kl = pescador.Streamer(_cycle, 'kl')
        mux2 = mux_class([gh, ij, kl], 3, rate=None)

        # And inspect the second mux
        samples2 = list(mux2(max_iter=6 * 10))
        count2 = collections.Counter(samples2)
        assert set(count2.keys()) == set('ghijkl')

        # Merge the muxes together.
        mux3 = mux_class([mux1, mux2], 2, rate=None)
        samples3 = list(mux3.iterate(max_iter=10000))
        count3 = collections.Counter(samples3)
        assert set('abcdefghijkl') == set(count3.keys())
        max_count, min_count = max(count3.values()), min(count3.values())
        assert (max_count - min_count) / max_count < 0.2


@pytest.mark.parametrize('mux_class', [
    functools.partial(pescador.mux.Mux,
                      with_replacement=False, revive=True),
    functools.partial(pescador.mux.PoissonMux, mode="single_active")],
    ids=["DeprecatedMux",
         "PoissonMux"])
class TestPoissonMux_SingleActive:
    @pytest.mark.parametrize('n_streams', [1, 2, 4])
    @pytest.mark.parametrize('n_samples', [512])
    @pytest.mark.parametrize('k', [1, 2, 4])
    @pytest.mark.parametrize('rate', [1.0, 2.0, 4.0])
    def test_mux_single_active(self, mux_class, n_streams, n_samples, k, rate):
        streamers = [pescador.Streamer(T.finite_generator, 10)
                     for _ in range(n_streams)]

        mux = mux_class(streamers, k, rate=rate)
        estimate = list(mux.iterate(n_samples))

        # Make sure we get the right number of samples
        # This is highly improbable when revive=False
        assert len(estimate) == n_samples

    def test_mux_of_muxes_itered(self, mux_class):
        # Check on Issue #79
        abc = pescador.Streamer('abc')
        xyz = pescador.Streamer('xyz')
        mux1 = mux_class([abc, xyz], 10, rate=None,
                         prune_empty_streams=False, random_state=135)
        samples1 = mux1.iterate(max_iter=1000)
        count1 = collections.Counter(samples1)
        assert set('abcxyz') == set(count1.keys())

        n123 = pescador.Streamer('123')
        n456 = pescador.Streamer('456')
        mux2 = mux_class([n123, n456], 10, rate=None,
                         prune_empty_streams=False,
                         random_state=246)
        samples2 = mux2.iterate(max_iter=1000)
        count2 = collections.Counter(samples2)
        assert set('123456') == set(count2.keys())

        # Note that (random_state=987, k_active=2) fails.
        mux3 = mux_class([mux1, mux2], 10, rate=None,
                         prune_empty_streams=False,
                         random_state=987)
        samples3 = mux3.iterate(max_iter=1000)
        count3 = collections.Counter(samples3)
        assert set('abcxyz123456') == set(count3.keys())

    def test_mux_of_muxes_single(self, mux_class):
        # Check on Issue #79
        abc = pescador.Streamer('abc')
        xyz = pescador.Streamer('xyz')
        mux1 = mux_class([abc, xyz], 2, rate=None,
                         prune_empty_streams=False)

        n123 = pescador.Streamer('123')
        n456 = pescador.Streamer('456')
        mux2 = mux_class([n123, n456], 2, rate=None,
                         prune_empty_streams=False)

        mux3 = mux_class([mux1, mux2], 2, rate=None,
                         prune_empty_streams=False)
        samples3 = list(mux3.iterate(max_iter=10000))
        count3 = collections.Counter(samples3)
        assert set('abcxyz123456') == set(count3.keys())

    def test_critical_mux_of_rate_limited_muxes(self, mux_class):
        # Check on Issue #79

        ab = pescador.Streamer(_choice, 'ab')
        cd = pescador.Streamer(_choice, 'cd')
        ef = pescador.Streamer(_choice, 'ef')
        mux1 = mux_class([ab, cd, ef], 2, rate=2)

        gh = pescador.Streamer(_choice, 'gh')
        ij = pescador.Streamer(_choice, 'ij')
        kl = pescador.Streamer(_choice, 'kl')

        mux2 = mux_class([gh, ij, kl], 2, rate=2)

        mux3 = mux_class([mux1, mux2], 2, rate=None)
        samples = list(mux3.iterate(max_iter=10000))
        count = collections.Counter(samples)
        max_count, min_count = max(count.values()), min(count.values())
        assert (max_count - min_count) / max_count < 0.2
        assert set('abcdefghijkl') == set(count.keys())

    def test_restart_mux(self, mux_class):
        s1 = pescador.Streamer('abc')
        s2 = pescador.Streamer('def')
        mux = mux_class([s1, s2], 2, rate=None, random_state=1234)
        assert len(list(mux(max_iter=100))) == len(list(mux(max_iter=100)))

    def test_critical_mux(self, mux_class):
        # Check on Issue #80
        chars = 'abcde'
        streamers = [pescador.Streamer(x * 5) for x in chars]
        mux = mux_class(streamers, len(chars), rate=None,
                        prune_empty_streams=False, random_state=135)
        samples = mux.iterate(max_iter=1000)
        print(collections.Counter(samples))

    # Note: `timeout` is necessary to break the infinite loop in the
    # event a change causes this test to fail.
    @pytest.mark.timeout(2.0)
    def test_mux_inf_loop(self, mux_class):
        s1 = pescador.Streamer([])
        s2 = pescador.Streamer([])
        mux = mux_class([s1, s2], 2, rate=None, random_state=1234)

        assert len(list(mux(max_iter=100))) == 0

    def test_mux_stacked_uniform_convergence(self, mux_class):
        """This test is designed to check that boostrapped streams of data
        (Streamer subsampling, rate limiting) cascaded through multiple
        multiplexors converges in expectation to a flat, uniform sample of the
        stream directly.
        """
        ab = pescador.Streamer(_choice, 'ab')
        cd = pescador.Streamer(_choice, 'cd')
        ef = pescador.Streamer(_choice, 'ef')
        mux1 = mux_class([ab, cd, ef], 2, rate=2, random_state=1357)

        gh = pescador.Streamer(_choice, 'gh')
        ij = pescador.Streamer(_choice, 'ij')
        kl = pescador.Streamer(_choice, 'kl')

        mux2 = mux_class([gh, ij, kl], 2, rate=2, random_state=2468)

        stacked_mux = mux_class([mux1, mux2], 2, rate=None,
                                random_state=12345)

        max_iter = 1000
        chars = 'abcdefghijkl'
        samples = list(stacked_mux.iterate(max_iter=max_iter))
        counter = collections.Counter(samples)
        assert set(chars) == set(counter.keys())

        counts = np.asarray(list(counter.values()))

        # Check that the pvalue for the chi^2 test is at least 0.95
        test = scipy.stats.chisquare(counts)
        assert test.pvalue >= 0.95


class TestShuffledMux:
    def test_shuffled_mux_simple(self):
        "Test that `ShuffledMux` samples from all provided streams"
        to_generate = ['a', 'b', 'c', 'd', 'e']
        streams = [pescador.Streamer(_cycle, x) for x in to_generate]
        mux = pescador.ShuffledMux(streams, random_state=10)

        samples = list(mux.iterate(max_iter=1000))
        counter = collections.Counter(samples)

        # Test that there is [a, b, c] in the set
        assert set(counter.keys()) == set(to_generate)

        # Test that the statistics line up with expected.
        for i, key in enumerate(to_generate):
            np.testing.assert_approx_equal(counter[key] / len(samples),
                                           mux.weights[i],
                                           significant=1)

    def test_shuffled_mux_with_empty_streams(self):
        """Tests that empty streams are dropped, and that `ShuffledMux`
        actually restarts given finite streams.
        """
        things_to_generate = [
            "a", [], "b", [], [], "c"
        ]
        streamers = [pescador.Streamer(x) for x in things_to_generate]
        mux = pescador.mux.ShuffledMux(streamers, random_state=1234)

        samples = []
        # Iterate over the list manually so we can introspect the state
        # in the middle.
        for i, sample in enumerate(mux.iterate(30)):
            samples.append(sample)

            assert mux.streams_ is not None
            assert mux.weight_norm_ > 0

            # Check to make sure that the empty streams got their probabilities
            # set to 0 when they didn't produce any data.
            if i == 29:
                assert mux.stream_weights_[1] == 0
                assert mux.stream_weights_[3] == 0
                assert mux.stream_weights_[4] == 0

        assert len(samples) == 30

        counter = collections.Counter(samples)
        assert set(counter.keys()) == set(['a', 'b', 'c'])
        for key in ['a', 'b', 'c']:
            assert counter[key] > 0

    def test_shuffled_mux_weights(self):
        "When sampling with weights, do the statistics line up?"
        a = pescador.Streamer(_cycle, 'a')
        b = pescador.Streamer(_cycle, 'b')
        c = pescador.Streamer(_cycle, 'c')

        weights = [.6, .3, .1]
        mux = pescador.ShuffledMux([a, b, c], weights=weights, random_state=10)

        samples = list(mux.iterate(max_iter=1000))
        counter = collections.Counter(samples)

        # Test that there is [a, b, c] in the set
        assert set(counter.keys()) == set(['a', 'b', 'c'])

        # Test the statistics on the counts.
        # Does the sampling approximately match the weights?
        for i, key in enumerate(['a', 'b', 'c']):
            np.testing.assert_approx_equal(counter[key] / len(samples),
                                           weights[i],
                                           significant=1)


class TestRoundRobinMux:
    """The RoundRobinMux is guaranteed to reproduce samples in the
    same order as original streams.
    """

    def test_roundrobin_mux_simple(self):
        ab = pescador.Streamer('ab')
        cde = pescador.Streamer('cde')
        fghi = pescador.Streamer('fghi')
        mux = pescador.mux.RoundRobinMux([ab, cde, fghi], 'exhaustive')
        assert "".join(list(mux.iterate())) == "acfbdgehi"

    def test_rr_with_empty_streams(self):
        things_to_generate = [
            "aa", [], "bb", [], [], [], "cccc"
        ]
        streamers = [pescador.Streamer(x) for x in things_to_generate]
        mux = pescador.mux.RoundRobinMux(streamers, 'exhaustive')
        result = "".join(list(mux.iterate()))
        assert result == "abcabccc"

    def test_rr_basic_cycle(self):
        a = pescador.Streamer('a')
        b = pescador.Streamer('bb')
        mux = pescador.mux.RoundRobinMux([a, b], 'cycle')
        assert "".join(list(mux.iterate(7))) == "abbabba"

    def test_rr_permuted_cycle(self):
        a = pescador.Streamer('a')
        b = pescador.Streamer('bb')
        empty = pescador.Streamer([])
        c = pescador.Streamer('c')
        mux = pescador.mux.RoundRobinMux([a, b, empty, c], 'permuted_cycle')

        result = list(mux.iterate(12))
        counts = collections.Counter(result)
        assert len(counts) == 3
        assert counts['a'] == 3
        assert counts['b'] == 6
        assert counts['c'] == 3

    def test_rr_multiple_copies(self):
        ab = pescador.Streamer('ab')
        cde = pescador.Streamer('cde')
        fghi = pescador.Streamer('fghi')
        mux = pescador.mux.RoundRobinMux([ab, cde, fghi], 'exhaustive')

        gen1 = mux.iterate(3)
        gen2 = mux.iterate()  # n == 9

        # No streamers should be active until we actually start the generators
        assert mux.active == 0

        # grab one sample each to make sure we've actually started the
        # generator
        _ = next(gen1)
        _ = next(gen2)
        assert mux.active == 2

        # the first one should die after two more samples
        result1 = list(gen1)
        assert "".join(result1) == "cf"
        assert len(result1) == 2
        assert mux.active == 1

        # The second should die after 6
        result2 = list(gen2)
        assert "".join(result2) == "cfbdgehi"
        assert len(result2) == 8
        assert mux.active == 0


class TestChainMux:
    @pytest.mark.parametrize('mode', [
        "exhaustive", "cycle",
        pytest.mark.xfail("foo")
    ])
    def test_modes(self, mode):
        a = pescador.Streamer("abc")
        b = pescador.Streamer("def")
        mux = pescador.mux.ChainMux([a, b],
                                    mode="exhaustive")
        result = list(mux.iterate())
        assert len(result) > 0

    def test_chain_mux_exhaustive(self):
        a = pescador.Streamer("abc")
        b = pescador.Streamer("def")
        mux = pescador.mux.ChainMux([a, b],
                                    mode="exhaustive")
        assert "".join(list(mux.iterate())) == "abcdef"
        # Make sure it's the same as itertools.chain
        assert list(mux.iterate()) == list(
            itertools.chain(a.iterate(), b.iterate()))

    def test_chain_mux_exhaustive_many(self):
        a = pescador.Streamer("a")
        b = pescador.Streamer("b")
        c = pescador.Streamer("c")
        d = pescador.Streamer("d")
        e = pescador.Streamer("e")
        f = pescador.Streamer("f")
        g = pescador.Streamer("g")

        mux = pescador.mux.ChainMux([a, b, c, d, e, f, g],
                                    mode="exhaustive")
        assert "".join(list(mux.iterate())) == "abcdefg"

    def test_chain_mux_cycle(self):
        a = pescador.Streamer("abc")
        b = pescador.Streamer("def")
        mux = pescador.mux.ChainMux([a, b],
                                    mode="cycle")
        assert "".join(list(mux.iterate(max_iter=12))) == "abcdefabcdef"

    def test_chain_streamer_of_streams(self):
        """If you want to pass parameters to your generator function,
        you have to do it with a streamer.
        """
        def stream_gen(n, source_letters):
            """
            Parameters
            ----------
            n : how many to stream.

            source_letters : list of things to stream.
            """
            for char in source_letters:
                yield pescador.Streamer(char * n)

        streamers = pescador.Streamer(stream_gen, 10, "abcde")
        mux = pescador.mux.ChainMux(streamers, mode="exhaustive")
        result = "".join(list(mux.iterate()))
        assert len(result) == 50
        assert result == "{}{}{}{}{}".format(
            'a' * 10, 'b' * 10, 'c' * 10, 'd' * 10, 'e' * 10)

    def test_chain_empty_streamer_of_streams(self):
        def stream_gen():
            return iter(())
        # stream_gen doesn't contain a yield statement, so we have to
        # create the generator here
        streamers = stream_gen()
        mux = pescador.mux.ChainMux(streamers, mode="exhaustive")
        result = "".join(list(mux.iterate()))
        assert len(result) == 0
        assert result == ''

    def test_chain_generatorfn_with_empty_streams(self):
        def stream_gen():
            things_to_generate = [
                "aa", [], "bb", [], [], [], "cccc"
            ]
            for item in things_to_generate:
                yield pescador.Streamer(item)

        mux = pescador.mux.ChainMux(stream_gen, mode="exhaustive")
        result = "".join(list(mux.iterate()))
        assert len(result) == 8
        assert result == "aabbcccc"

    def test_chain_generator_with_multiple_copies(self):
        def stream_gen():
            things_to_generate = [
                "a", "bb", [], "ccc"
            ]
            for item in things_to_generate:
                yield pescador.Streamer(item)

        mux = pescador.mux.ChainMux(stream_gen, mode="exhaustive")

        gen1 = mux.iterate(3)
        gen2 = mux.iterate()  # n == 6

        # No streamers should be active until we actually start the generators
        assert mux.active == 0

        # grab one sample each to make sure we've actually started the
        # generator
        _ = next(gen1)
        _ = next(gen2)
        assert mux.active == 2

        # the first one should die after two more samples
        result1 = list(gen1)
        assert "".join(result1) == "bb"
        assert len(result1) == 2
        assert mux.active == 1

        # The second should die after 6
        result2 = list(gen2)
        assert "".join(result2) == "bbccc"
        assert len(result2) == 5
        assert mux.active == 0
