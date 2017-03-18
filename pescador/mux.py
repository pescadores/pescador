#!/usr/bin/env python
'''Stream multiplexing'''
import six
import numpy as np

from . import core
from .exceptions import PescadorError


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
    '''

    def __init__(self, seed_pool, k,
                 lam=256.0, pool_weights=None, with_replacement=True,
                 prune_empty_seeds=True, revive=False,
                 random_state=None):
        """Given an array of Streamer objects, do the following:

        1. Select ``k`` seeds at random to activate
        2. Assign each activated seed a sample count ~ Poisson(lam)
        3. Yield samples from the streams by randomly multiplexing
           from the active set.
        4. When a stream is exhausted, select a new one from the pool.

        Parameters
        ----------
        seed_pool : iterable of Streamer
            The collection of Streamer objects

        k : int > 0
            The number of streams to keep active at any time.

        lam : float > 0 or None
            Rate parameter for the Poisson distribution governing sample counts
            for individual streams.
            If ``None``, sample infinitely from each stream.

        pool_weights : np.ndarray or None
            Optional weighting for ``seed_pool``.
            If ``None``, then weights are assumed to be uniform.
            Otherwise, ``pool_weights[i]`` defines the sampling proportion
            of ``seed_pool[i]``.

            Must have the same length as ``seed_pool``.

        with_replacement : bool
            Sample Streamers with replacement.  This allows a single stream to
            be used multiple times (even simultaneously).
            If ``False``, then each Streamer is consumed at most once and never
            revisited.

        prune_empty_seeds : bool
            Disable seeds from the pool that produced no data.
            If ``True``, Streamers that previously produced no data are never
            revisited.
            Note that this may be undesireable for streams where past emptiness
            may not imply future emptiness.

        revive: bool
            If ``with_replacement`` is ``False``, setting ``revive=True``
            will re-insert previously exhausted seeds into the candidate set.

            This configuration allows a seed to be active at most once at any
            time.

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.
        """
        self.seed_pool = seed_pool
        self.n_seeds = len(seed_pool)
        self.k = k
        self.lam = lam
        self.pool_weights = pool_weights
        self.with_replacement = with_replacement
        self.prune_empty_seeds = prune_empty_seeds
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

        if not self.n_seeds:
            raise PescadorError('Cannot mux an empty seed-pool')

        # Set up the sampling distribution over streams
        self.seed_distribution = 1. / self.n_seeds * np.ones(self.n_seeds)

        if self.pool_weights is None:
            self.pool_weights = self.seed_distribution.copy()

        self.pool_weights = np.atleast_1d(self.pool_weights)

        if len(self.pool_weights) != len(self.seed_pool):
            raise PescadorError('pool_weights must be the same '
                                'length as seed_pool')

        if not (self.pool_weights > 0.0).any():
            raise PescadorError('pool_weights must contain at least '
                                'one positive value')

        self.pool_weights /= np.sum(self.pool_weights)

    def activate(self):
        """Activates the seed pool"""
        # Instantiate the pool
        self.streams_ = [None] * self.k

        self.stream_weights_ = np.zeros(self.k)
        self.stream_counts_ = np.zeros(self.k, dtype=int)
        self.stream_idxs_ = np.zeros(self.k, dtype=int)

        for idx in range(self.k):

            if not (self.seed_distribution > 0).any():
                break

            self.stream_idxs_[idx] = self.rng.choice(
                self.n_seeds, p=self.seed_distribution)
            self.streams_[idx], self.stream_weights_[idx] = (
                self.__new_seed(self.stream_idxs_[idx]))

        self.weight_norm_ = np.sum(self.stream_weights_)

    def deactivate(self):
        self.streams_ = None
        self.stream_weights_ = None
        self.stream_counts_ = None
        self.stream_idxs_ = None
        self.weight_norm_ = None

    def generate(self, max_batches=None):
        with core.StreamActivator(self):

            # Main sampling loop
            n = 0

            if max_batches is None:
                max_batches = np.inf

            while n < max_batches and self.weight_norm_ > 0.0:
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

                    if (self.prune_empty_seeds and
                            self.stream_counts_[idx] == 0):
                        # If we're disabling empty seeds, see if this stream
                        # produced data
                        self.seed_distribution[self.stream_idxs_[idx]] = 0.0

                    if self.revive and not self.with_replacement:
                        # If we need to revive a seed, give it the max
                        # current probability
                        if self.seed_distribution.any():
                            self.seed_distribution[self.stream_idxs_[idx]] = (
                                np.max(self.seed_distribution))
                        else:
                            self.seed_distribution[self.stream_idxs_[idx]] = 1.0

                    if (self.seed_distribution > 0).any():
                        # Replace it and move on if there are still seeds
                        # in the pool.
                        self.seed_distribution[:] /= np.sum(
                            self.seed_distribution)

                        self.stream_idxs_[idx] = self.rng.choice(
                            self.n_seeds, p=self.seed_distribution)

                        self.streams_[idx], self.stream_weights_[idx] = (
                            self.__new_seed(self.stream_idxs_[idx]))

                        self.stream_counts_[idx] = 0

                    else:
                        # Otherwise, this one's exhausted.
                        # Set its probability to 0
                        self.stream_weights_[idx] = 0.0

                    self.weight_norm_ = np.sum(self.stream_weights_)

    def __new_seed(self, idx):
        '''Randomly select and create a stream from the pool.

        Parameters
        ----------
        idx : int
            The seed index to replace
        '''
        if len(self.seed_pool) != len(self.pool_weights):
            raise PescadorError('seed_pool must have the same '
                                'length as pool_weights')

        if len(self.seed_pool) != len(self.seed_distribution):
            raise PescadorError('seed_pool must have the same '
                                'length as seed_distribution')

        # instantiate
        if self.lam is not None:
            n_stream = 1 + self.rng.poisson(lam=self.lam)
        else:
            n_stream = None

        # If we're sampling without replacement, zero this one out
        if not self.with_replacement:
            self.seed_distribution[idx] = 0.0

            if (self.seed_distribution > 0).any():
                self.seed_distribution[:] /= np.sum(self.seed_distribution)

        return (self.seed_pool[idx].generate(max_batches=n_stream),
                self.pool_weights[idx])
