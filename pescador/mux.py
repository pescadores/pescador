import six
import numpy as np

import pescador.core


class Mux(pescador.core.Streamer):
    '''Stochastic multiplexor for generator seeds.'''

    def __init__(self, seed_pool, k,
                 lam=256.0, pool_weights=None, with_replacement=True,
                 prune_empty_seeds=True, revive=False):
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
        """
        self.seed_pool = seed_pool
        self.n_seeds = len(seed_pool)
        self.k = k
        self.lam = lam
        self.pool_weights = pool_weights
        self.with_replacement = with_replacement
        self.prune_empty_seeds = prune_empty_seeds
        self.revive = revive

        self.reset()

        if not self.n_seeds:
            raise RuntimeError('Cannot mux an empty seed-pool')

        # Set up the sampling distribution over streams
        self.seed_distribution = 1. / self.n_seeds * np.ones(self.n_seeds)

        if self.pool_weights is None:
            self.pool_weights = self.seed_distribution.copy()

        self.pool_weights = np.atleast_1d(self.pool_weights)

        assert len(self.pool_weights) == len(self.seed_pool)
        assert (self.pool_weights > 0.0).any()
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

            self.stream_idxs_[idx] = np.random.choice(
                self.n_seeds, p=self.seed_distribution)
            self.streams_[idx], self.stream_weights_[idx] = generate_new_seed(
                self.stream_idxs_[idx], self.seed_pool, self.pool_weights,
                self.seed_distribution, self.lam, self.with_replacement)

        self.weight_norm_ = np.sum(self.stream_weights_)

    def reset(self):
        self.streams_ = None
        self.stream_weights_ = None
        self.stream_counts_ = None
        self.stream_idxs_ = None
        self.weight_norm_ = None

    def generate(self, max_batches=None):
        self.activate()

        # Main sampling loop
        n = 0

        if max_batches is None:
            max_batches = np.inf

        while n < max_batches and self.weight_norm_ > 0.0:
            # Pick a stream from the active set
            idx = np.random.choice(self.k, p=(self.stream_weights_ /
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

                if self.prune_empty_seeds and self.stream_counts_[idx] == 0:
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
                    self.seed_distribution[:] /= np.sum(self.seed_distribution)

                    self.stream_idxs_[idx] = np.random.choice(
                        self.n_seeds, p=self.seed_distribution)

                    self.streams_[idx], self.stream_weights_[idx] = (
                        generate_new_seed(self.stream_idxs_[idx],
                                          self.seed_pool, self.pool_weights,
                                          self.seed_distribution, self.lam,
                                          self.with_replacement))

                    self.stream_counts_[idx] = 0

                else:
                    # Otherwise, this one's exhausted.
                    # Set its probability to 0
                    self.stream_weights_[idx] = 0.0

                self.weight_norm_ = np.sum(self.stream_weights_)

        self.reset()


def generate_new_seed(idx, pool, weights, distribution, lam=256.0,
                      with_replacement=True):
    '''Randomly select and create a stream from the pool.

    Parameters
    ----------
    idx : int

    pool : iterable of Streamer
        The collection of Streamer objects

    weights : np.ndarray or None
        Defines the stream sample weight of each ``pool[i]``.

        Must have the same length as ``pool``.

    distribution : np.ndarray
        Defines the probability of selecting the item '`pool[i]``.

        Notes:
        1. Must have the same length as ``pool``.
        2. ``distribution`` will be modified in-place when
        with_replacement=False.

    lam : float > 0 or None
        Rate parameter for the Poisson distribution governing sample counts
        for individual streams.
        If ``None``, sample infinitely from each stream.

    with_replacement : bool
        Sample Streamers with replacement.  This allows a single stream to be
        used multiple times (even simultaneously).
        If ``False``, then each Streamer is consumed at most once and never
        revisited.
    '''
    assert len(pool) == len(weights) == len(distribution)
    # instantiate
    if lam is not None:
        n_stream = 1 + np.random.poisson(lam=lam)
    else:
        n_stream = None

    # If we're sampling without replacement, zero this one out
    if not with_replacement:
        distribution[idx] = 0.0

        if (distribution > 0).any():
            distribution[:] /= np.sum(distribution)

    return pool[idx].generate(max_batches=n_stream), weights[idx]
