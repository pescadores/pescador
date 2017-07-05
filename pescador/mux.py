#!/usr/bin/env python
'''Stream multiplexing'''
import six
import numpy as np

from . import core
from .exceptions import PescadorError
from .util import Deprecated, rename_kw


class Mux(core.Streamer):
    '''Stochastic multiplexor for Streamers

    Examples
    --------
    >>> # Create a collection of streamers
    >>> seeds = [pescador.Streamer(my_generator) for i in range(10)]
    >>> # Multiplex them together into a single streamer
    >>> # Use at most 3 streams at once
    >>> mux = pescador.Mux(seeds, k=3)
    >>> for batch in mux():
    ...     MY_FUNCTION(batch)

    Mux([stream, range(8), stream2])
    '''

    def __init__(self, streamers, k,
                 rate=256.0, weights=None, with_replacement=True,
                 prune_empty_streams=True, revive=False,
                 random_state=None,
                 seed_pool=Deprecated(),
                 lam=Deprecated(),
                 pool_weights=Deprecated(),
                 prune_empty_seeds=Deprecated()):
        """Given an array (pool) of streamer types, do the following:

        1. Select ``k`` streams at random to iterate from
        2. Assign each activated stream a sample count ~ Poisson(lam)
        3. Yield samples from the streams by randomly multiplexing
           from the active set.
        4. When a stream is exhausted, select a new one from `streamers`.

        Parameters
        ----------
        streamers : iterable of streamers
            The collection of streamer-type objects

        k : int > 0
            The number of streams to keep active at any time.

        rate : float > 0 or None
            Rate parameter for the Poisson distribution governing sample counts
            for individual streams.
            If ``None``, sample infinitely from each stream.

        weights : np.ndarray or None
            Optional weighting for ``streamers``.
            If ``None``, then weights are assumed to be uniform.
            Otherwise, ``weights[i]`` defines the sampling proportion
            of ``streamers[i]``.

            Must have the same length as ``streamers``.

        with_replacement : bool
            Sample streamers with replacement.  This allows a single stream to
            be used multiple times (even simultaneously).
            If ``False``, then each streamer is consumed at most once and never
            revisited.

        prune_empty_streams : bool
            Disable streamers that produce no data.
            If ``True``, streamers that previously produced no data are never
            revisited.
            Note:
            1. This may be undesireable for streams where past emptiness
            may not imply future emptiness.
            2. Failure to prune truly empty streams with `revive=True` can
            result in infinite looping behavior. Disable with caution.

        revive: bool
            If ``with_replacement`` is ``False``, setting ``revive=True``
            will re-insert previously exhausted streams into the candidate set.

            This configuration allows a stream to be active at most once at any
            time.

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.

        seed_pool : iterable of streamers
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `streamers` parameter instead.
                The `seed_pool` parameter will be removed in pescador 2.0.

        lam : float > 0.0
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `rate` parameter instead.
                The `lam` parameter will be removed in pescador 2.0.

        pool_weights : np.ndarray or None
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `weights` parameter instead.
                The `pool_weights` parameter will be removed in pescador 2.0.

        prune_empty_seeds : bool
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `prune_empty_streams` parameter instead.
                The `prune_empty_seeds` parameter will be removed in
                pescador 2.0.
        """
        streamers = rename_kw('seed_pool', seed_pool,
                              'streamers', streamers,
                              '1.1', '2.0')
        rate = rename_kw('lam', lam,
                         'rate', rate,
                         '1.1', '2.0')
        weights = rename_kw('pool_weights', pool_weights,
                            'weights', weights,
                            '1.1', '2.0')
        prune_empty_streams = rename_kw(
            'prune_empty_seeds', prune_empty_seeds,
            'prune_empty_streams', prune_empty_streams,
            '1.1', '2.0')
        self.streamers = streamers
        self.n_streams = len(streamers)
        self.k = k
        self.rate = rate
        self.weights = weights
        self.with_replacement = with_replacement
        self.prune_empty_streams = prune_empty_streams
        self.revive = revive

        self.deactivate()

        if random_state is None:
            self.rng = np.random
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            raise PescadorError('Invalid random_state={}'.format(random_state))

        if not self.n_streams:
            raise PescadorError('Cannot mux an empty collection')

        if self.weights is None:
            self.weights = 1. / self.n_streams * np.ones(self.n_streams)

        self.weights = np.atleast_1d(self.weights)

        if len(self.weights) != len(self.streamers):
            raise PescadorError('`weights` must be the same '
                                'length as `streamers`')

        if not (self.weights > 0.0).any():
            raise PescadorError('`weights` must contain at least '
                                'one positive value')

        self.weights /= np.sum(self.weights)

    def activate(self):
        """Activates a number of streams"""
        self.distribution_ = 1. / self.n_streams * np.ones(self.n_streams)
        self.valid_streams_ = np.ones(self.n_streams, dtype=bool)

        self.streams_ = [None] * self.k

        self.stream_weights_ = np.zeros(self.k)
        self.stream_counts_ = np.zeros(self.k, dtype=int)
        # Array of pointers into `self.streamers`
        self.stream_idxs_ = np.zeros(self.k, dtype=int)

        for idx in range(self.k):

            if not (self.distribution_ > 0).any():
                break

            self.stream_idxs_[idx] = self.rng.choice(
                self.n_streams, p=self.distribution_)
            self.streams_[idx], self.stream_weights_[idx] = (
                self.__new_stream(self.stream_idxs_[idx]))

        self.weight_norm_ = np.sum(self.stream_weights_)

    def deactivate(self):
        self.streams_ = None
        self.stream_weights_ = None
        self.stream_counts_ = None
        self.stream_idxs_ = None
        self.weight_norm_ = None

    def iterate(self, max_iter=None):
        with core.StreamActivator(self):

            # Main sampling loop
            n = 0

            if max_iter is None:
                max_iter = np.inf

            while n < max_iter and self.weight_norm_ > 0.0:
                # Pick a stream from the active set
                idx = self.rng.choice(self.k, p=(self.stream_weights_ /
                                                 self.weight_norm_))

                # Can we sample from it?
                try:
                    # Then yield the sample
                    yield six.advance_iterator(self.streams_[idx])

                    # Increment the sample counter
                    n += 1
                    self.stream_counts_[idx] += 1

                except StopIteration:
                    # Oops, this one's exhausted.

                    if (self.prune_empty_streams and
                            self.stream_counts_[idx] == 0):
                        # If we're disabling empty seeds, see if this stream
                        # produced data; if it didn't, turn it off.
                        self.distribution_[self.stream_idxs_[idx]] = 0.0
                        self.valid_streams_[self.stream_idxs_[idx]] = False

                    if self.revive and not self.with_replacement:
                        # If we need to revive a seed, give it the max
                        # current probability
                        if self.distribution_.any():
                            self.distribution_[self.stream_idxs_[idx]] = (
                                np.max(self.distribution_))
                        else:
                            self.distribution_[self.stream_idxs_[idx]] = 1.0

                    if (self.distribution_ > 0).any():
                        # Replace it and move on if there are still seeds
                        # in the pool.
                        self.distribution_[:] /= np.sum(self.distribution_)

                        self.stream_idxs_[idx] = self.rng.choice(
                            self.n_streams, p=self.distribution_)

                        self.streams_[idx], self.stream_weights_[idx] = (
                            self.__new_stream(self.stream_idxs_[idx]))

                        self.stream_counts_[idx] = 0

                    else:
                        # Otherwise, this one's exhausted.
                        # Set its probability to 0
                        self.stream_weights_[idx] = 0.0

                    self.weight_norm_ = np.sum(self.stream_weights_)

                # If everything has been pruned, kill the while loop
                if not self.valid_streams_.any():
                    break

    def __new_stream(self, idx):
        '''Randomly select and create a stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        if len(self.streamers) != len(self.weights):
            raise PescadorError('`streamers` must have the same '
                                'length as `weights`')

        if len(self.streamers) != len(self.distribution_):
            raise PescadorError('`streamers` must have the same '
                                'length as `distribution`')

        # instantiate
        if self.rate is not None:
            n_stream = 1 + self.rng.poisson(lam=self.rate)
        else:
            n_stream = None

        # If we're sampling without replacement, zero this one out
        if not self.with_replacement:
            self.distribution_[idx] = 0.0

            if (self.distribution_ > 0).any():
                self.distribution_[:] /= np.sum(self.distribution_)

        return (self.streamers[idx].iterate(max_iter=n_stream),
                self.weights[idx])
