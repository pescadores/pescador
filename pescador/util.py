#!/usr/bin/env python
'''Utility functions for stream manipulations

.. autosummary::
    :toctree: generated/

    mux
    buffer_batch
    buffer_streamer
    batch_length

'''
import numpy as np
import six

__all__ = ['mux', 'buffer_batch', 'batch_length', 'buffer_streamer']


def batch_length(batch):
    '''Determine the number of samples in a batch.

    Parameters
    ----------
    batch : dict
        A batch dictionary.  Each value must implement `len`.
        All values must have the same `len`.

    Returns
    -------
    n : int >= 0 or None
        The number of samples in this batch.
        If the batch has no fields, n is None.

    Raises
    ------
    RuntimeError
        If some two values have unequal length
    '''
    n = None

    for value in six.itervalues(batch):
        if n is None:
            n = len(value)

        elif len(value) != n:
            raise RuntimeError('Unequal field lengths')

    return n


def mux(seed_pool, n_samples, k, lam=256.0, pool_weights=None,
        with_replacement=True, prune_empty_seeds=True):
    '''Stochastic multiplexor for generator seeds.

    Given an array of Streamer objects, do the following:

        1. Select ``k`` seeds at random to activate
        2. Assign each activated seed a sample count ~ Poisson(lam)
        3. Yield samples from the streams by randomly multiplexing
           from the active set.
        4. When a stream is exhausted, select a new one from the pool.

    Parameters
    ----------
    seed_pool : iterable of Streamer
        The collection of Streamer objects

    n_samples : int > 0 or None
        The number of samples to generate.
        If ``None``, sample indefinitely.

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
        Sample Streamers with replacement.  This allows a single stream to be
        used multiple times (even simultaneously).
        If ``False``, then each Streamer is consumed at most once and never
        revisited.

    prune_empty_seeds : bool
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

    pool_weights = np.atleast_1d(pool_weights)

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
            yield six.advance_iterator(streams[idx])

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


def buffer_batch(generator, buffer_size):
    '''Buffer an iterable of batches into larger (or smaller) batches

    Parameters
    ----------
    generator : iterable
        The generator to buffer

    buffer_size : int > 0
        The number of examples to retain per batch.

    Yields
    ------
    batch
        A batch of size at most `buffer_size`
    '''

    batches = []
    n = 0

    for x in generator:
        batches.append(x)
        n += batch_length(x)

        if n < buffer_size:
            continue

        batch, batches = __split_batches(batches, buffer_size)

        if batch is not None:
            yield batch
            batch = None
            n = 0

    # Run out the remaining samples
    while batches:
        batch, batches = __split_batches(batches, buffer_size)
        if batch is not None:
            yield batch


def buffer_streamer(streamer, buffer_size, *args, **kwargs):
    '''Buffer a stream of batches

    Parameters
    ----------
    streamer : pescador.Streamer
        The streamer object to buffer

    buffer_size : int > 0
        the number of examples to retain per batch

    Yields
    ------
    batch
        A batch of size at most `buffer_size`

    See Also
    --------
    buffer_batch

    '''
    for batch in buffer_batch(streamer.generate(*args, **kwargs),
                              buffer_size):
        yield batch


def __split_batches(batches, buffer_size):
    '''Split at most one batch off of a collection of batches.

    Parameters
    ----------
    batches : list
        List of batch objects

    buffer_size : int > 0 or None
        Size of the desired buffer.
        If None, the entire stream is exhausted.

    Returns
    -------
    batch, remaining_batches
        One batch of size up to buffer_size,
        and all remaining batches.

    '''

    batch_size = 0
    batch_data = []

    # First, pull off all the candidate batches
    while batches and (buffer_size is None or
                       batch_size < buffer_size):
        batch_data.append(batches.pop(0))
        batch_size += batch_length(batch_data[-1])

    # Merge the batches
    batch = dict()
    residual = dict()

    has_residual = False
    has_data = False

    for key in batch_data[0].keys():
        batch[key] = np.concatenate([data[key] for data in batch_data])

        residual[key] = batch[key][buffer_size:]

        if len(residual[key]):
            has_residual = True

        # Clip to the appropriate size
        batch[key] = batch[key][:buffer_size]

        if len(batch[key]):
            has_data = True

    if has_residual:
        batches.insert(0, residual)

    if not has_data:
        batch = None

    return batch, batches


def generate_new_seed(idx, pool, weights, distribution, lam=256.0,
                      with_replacement=True):
    '''Randomly select and create a stream from the pool.

    Parameters
    ----------
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
