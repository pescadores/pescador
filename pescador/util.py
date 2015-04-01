#!/usr/bin/env python
'''Utilities'''
import numpy as np
import scipy.sparse


__all__ = ['mux', 'buffer_data']


def mux(seed_pool, n_samples, k, lam=256.0, pool_weights=None,
        with_replacement=True, prune_empty_seeds=True):
    '''Stochastic multiplexor for generator seeds.

    Given an array of Streamer objects, do the following:

        1. Select ``k`` seeds at random to activate
        2. Assign each activated seed a sample count ~ Poisson(lam)
        3. Yield samples from the streams by randomly multiplexing
           from the active set.
        4. When a stream is exhausted, select a new one from the pool.

    :parameters:
        - seed_pool : iterable of Streamer
          The collection of Streamer objects

        - n_samples : int > 0 or None
          The number of samples to generate.
          If ``None``, sample indefinitely.

        - k : int > 0
          The number of streams to keep active at any time.

        - lam : float > 0 or None
          Rate parameter for the Poisson distribution governing sample counts
          for individual streams.
          If ``None``, sample infinitely from each stream.

        - pool_weights : np.ndarray or None
          Optional weighting for ``seed_pool``.
          If ``None``, then weights are assumed to be uniform.
          Otherwise, ``pool_weights[i]`` defines the sampling proportion
          of ``seed_pool[i]``.

          Must have the same length as ``seed_pool``.

        - with_replacement : bool
          Sample Streamers with replacement.  This allows a single stream to be
          used multiple times (even simultaneously).
          If ``False``, then each Streamer is consumed at most once and never
          revisited.

        - prune_empty_seeds : bool
          Disable seeds from the pool that produced no data.
          If ``True``, Streamers that previously produced no data are never
          revisited.
          Note that this may be undesireable for streams where past emptiness
          may not imply future emptiness.

    '''
    n_seeds = len(seed_pool)

    # Set up the sampling distribution over streams
    seed_distribution = 1./n_seeds * np.ones(n_seeds)

    if pool_weights is None:
        pool_weights = seed_distribution.copy()

    assert len(pool_weights) == len(seed_pool)
    assert (pool_weights > 0.0).any()
    pool_weights /= np.sum(pool_weights)

    # Instantiate the pool
    streams = [None] * k

    stream_weights = np.zeros(k)
    stream_counts = np.zeros(k, dtype=int)
    stream_idxs = np.zeros(k, dtype=int)
    for idx in range(k):

        if not (seed_distribution > 0).any():
            break

        stream_idxs[idx] = np.random.choice(n_seeds, p=seed_distribution)
        streams[idx], stream_weights[idx] = generate_new_seed(
            stream_idxs[idx], seed_pool, pool_weights, seed_distribution, lam,
            with_replacement)

    weight_norm = np.sum(stream_weights)

    # Main sampling loop
    n = 0

    if n_samples is None:
        n_samples = np.inf

    while n < n_samples and weight_norm > 0.0:
        # Pick a stream from the active set
        idx = np.random.choice(k, p=stream_weights / weight_norm)

        # Can we sample from it?
        try:
            # Then yield the sample
            yield streams[idx].next()

            # Increment the sample counter
            n += 1
            stream_counts[idx] += 1

        except StopIteration:
            # Oops, this one's exhausted.
            # If we're disabling empty seeds, see if this stream produced data.
            if prune_empty_seeds and stream_counts[idx] == 0:
                seed_distribution[stream_idxs[idx]] = 0.0
                if (seed_distribution > 0).any():
                    seed_distribution[:] /= np.sum(seed_distribution)
            # Replace it and move on if there are still kids in the pool.
            if (seed_distribution > 0).any():
                stream_idxs[idx] = np.random.choice(n_seeds,
                                                    p=seed_distribution)
                streams[idx], stream_weights[idx] = generate_new_seed(
                    stream_idxs[idx], seed_pool, pool_weights,
                    seed_distribution, lam, with_replacement)
                stream_counts[idx] = 0

            else:
                # Otherwise, this one's exhausted.  Set its probability to 0
                stream_weights[idx] = 0.0

            weight_norm = np.sum(stream_weights)


def buffer_data(data):
    """Determine whether the data is sparse or not, and buffer it accordingly.

    :parameters:
        - data : list of scipy.sparse or np.ndarray
            The data to buffer

    :returns:
        - buf : scipy.sparse.csr or np.ndarray
            If the input data was sparse, a sparse matrix of the data
            concatenated vertically.
            Otherwise, the data stacked vertically as a dense ndarray.
    """

    if scipy.sparse.issparse(data[0]):
        n = len(data)
        dimension = np.prod(data[0].shape)

        data_s = scipy.sparse.lil_matrix((n, dimension), dtype=data[0].dtype)

        for i in range(len(data)):
            idx = data[i].indices
            data_s[i, idx] = data[i][:, idx]

        return data_s.tocsr()

    else:
        return np.asarray(data)


def generate_new_seed(idx, pool, weights, distribution, lam=256.0,
                       with_replacement=True):
    '''Randomly select and create a stream from the pool.

    :parameters:
    - pool : iterable of Streamer
        The collection of Streamer objects

    - weights : np.ndarray or None
        Defines the stream sample weight of each ``pool[i]``.

        Must have the same length as ``pool``.

    - distribution : np.ndarray
        Defines the probability of selecting the item '`pool[i]``.

        Notes:
        1. Must have the same length as ``pool``.
        2. ``distribution`` will be modified in-place when
        with_replacement=False.

    - lam : float > 0 or None
        Rate parameter for the Poisson distribution governing sample counts
        for individual streams.
        If ``None``, sample infinitely from each stream.

    - with_replacement : bool
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

    return pool[idx].generate(max_items=n_stream), weights[idx]
