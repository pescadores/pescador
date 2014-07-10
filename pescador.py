#!/usr/bin/python
"""Utilities to facilitate out-of-core learning in sklearn"""

import collections
import numpy as np
import scipy

import sklearn.base


class Streamer(object):
    '''A wrapper class for reusable generators.

    :usage:
        >>> # make a generator
        >>> def my_generator(n):
                for i in range(n):
                    yield i
        >>> GS = Streamer(my_generator, 5)
        >>> for i in GS.generate():
                print i

        >>> # Or with a maximum number of items
        >>> for i in GS.generate(max_items=3):
                print i

    :parameters:
        - streamer : function or iterable
          Any generator function or iterable python object

        - *args, **kwargs
          Additional positional arguments or keyword arguments to pass
          through to ``generator()``

    :raises:
        - TypeError
          If ``streamer`` is not a function or an Iterable object.
    '''

    def __init__(self, streamer, *args, **kwargs):
        '''Initializer'''

        if not (hasattr(streamer, '__call__') or
                isinstance(streamer, collections.Iterable)):
            raise TypeError('`streamer` must be a generator function or Iterable')

        self.stream = streamer
        self.args = args
        self.kwargs = kwargs

    def generate(self, max_items=None):
        '''Instantiate the generator

        :parameters:
            - max_items : None or int > 0
              Maximum number of items to yield.
              If ``None``, exhaust the generator.

        :yields:
            - Items from the contained generator
        '''

        if max_items is None:
            max_items = np.inf

        # If it's a function, create the stream.
        # If it's iterable, use it directly.

        if hasattr(self.stream, '__call__'):
            my_stream = self.stream(*(self.args), **(self.kwargs))

        else:
            my_stream = self.stream

        for i, x in enumerate(my_stream):
            if i >= max_items:
                break
            yield x


def categorical_sample(weights):
    '''Sample from a categorical distribution.

    :parameters:
        - weights : np.array, shape=(n,)
          The distribution to sample from.
          Must be non-negative and sum to 1.0.

    :returns:
        - k : int in [0, n)
          The sample
    '''

    return np.flatnonzero(np.random.multinomial(1, weights))[0]


def mux(seed_pool, n_samples, k, lam=256.0, pool_weights=None,
        with_replacement=True):
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

    '''
    n_seeds = len(seed_pool)

    # Set up the sampling distribution over streams
    seed_distribution = 1./n_seeds * np.ones(n_seeds)

    if pool_weights is None:
        pool_weights = seed_distribution.copy()

    assert len(pool_weights) == len(seed_pool)
    assert (pool_weights > 0.0).all()
    pool_weights /= np.sum(pool_weights)

    # Instantiate the pool
    streams = [None] * k

    stream_weights = np.zeros(k)

    for idx in range(k):

        if not (seed_distribution > 0).any():
            break

        # how many samples for this stream?
        # pick a stream
        new_idx = categorical_sample(seed_distribution)

        # instantiate
        if lam is not None:
            n_stream = np.random.poisson(lam=lam)
        else:
            n_stream = None

        streams[idx] = seed_pool[new_idx].generate(max_items=n_stream)
        stream_weights[idx] = pool_weights[new_idx]

        # If we're sampling without replacement, zero this one out
        if not with_replacement:
            seed_distribution[new_idx] = 0.0

            if (seed_distribution > 0).any():
                seed_distribution[:] /= np.sum(seed_distribution)

    weight_norm = np.sum(stream_weights)

    # Main sampling loop
    n = 0

    if n_samples is None:
        n_samples = np.inf

    while n < n_samples and weight_norm > 0.0:
        # Pick a stream
        idx = categorical_sample(stream_weights / weight_norm)

        # Can we sample from it?
        try:
            # Then yield the sample
            yield streams[idx].next()

            # Increment the sample counter
            n = n + 1

        except StopIteration:
            # Oops, this one's exhausted.  Replace it and move on.

            # Are there still kids in the pool?  Okay.
            if (seed_distribution > 0).any():

                new_idx = categorical_sample(pool_weights)

                if lam is not None:
                    n_stream = np.random.poisson(lam=lam)
                else:
                    n_stream = None

                streams[idx] = seed_pool[new_idx].generate(max_items=n_stream)

                stream_weights[idx] = pool_weights[new_idx]

                # If we're sampling without replacement, zero out this one out
                if not with_replacement:
                    seed_distribution[new_idx] = 0.0

                    if (seed_distribution > 0).any():
                        seed_distribution[:] /= np.sum(seed_distribution)

            else:
                # Otherwise, this one's exhausted.  Set its probability to 0
                stream_weights[idx] = 0.0

            weight_norm = np.sum(stream_weights)


def stream_fit(estimator, data_sequence, batch_size=100, max_steps=None,
               **kwargs):
    '''Fit a model to a generator stream.

    :parameters:
      - estimator : sklearn.base.BaseEstimator
        The model object.  Must implement ``partial_fit()``

      - data_sequence : generator
        A generator that yields samples

      - batch_size : int
        Maximum number of samples to buffer before updating the model

      - max_steps : int or None
        If ``None``, run until the stream is exhausted.
        Otherwise, run until at most ``max_steps`` examples
        have been processed.
    '''

    # Is this a supervised or unsupervised learner?
    supervised = isinstance(estimator, sklearn.base.ClassifierMixin)

    # Does the learner support partial fit?
    assert hasattr(estimator, 'partial_fit')

    def _matrixify(data):
        """Determine whether the data is sparse or not, act accordingly"""

        if scipy.sparse.issparse(data[0]):
            n = len(data)
            dimension = np.prod(data[0].shape)

            data_s = scipy.sparse.lil_matrix((n, dimension),
                                             dtype=data[0].dtype)

            for i in range(len(data)):
                idx = data[i].indices
                data_s[i, idx] = data[i][:, idx]

            return data_s.tocsr()
        else:
            return np.asarray(data)

    def _run(data, supervised):
        """Wrapper function to partial_fit()"""

        if supervised:
            args = [_matrixify(datum) for datum in zip(*data)]
        else:
            args = [_matrixify(data)]

        estimator.partial_fit(*args, **kwargs)

    buf = []
    for i, x_new in enumerate(data_sequence):
        buf.append(x_new)

        # We've run too far, stop
        if max_steps is not None and i > max_steps:
            break

        # Buffer is full, do an update
        if len(buf) == batch_size:
            _run(buf, supervised)
            buf = []

    # Update on whatever's left over
    if len(buf) > 0:
        _run(buf, supervised)
