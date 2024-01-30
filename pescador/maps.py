#!/usr/bin/env python
"""Map functions perform operations on a stream.

Important note: map functions return a *generator*, not another
Streamer, so if you need it to behave like a Streamer, you have to wrap
the function in a Streamer again.

.. autosummary::
    :toctree: generated/

    buffer_stream
    tuples
    keras_tuples
    cache
"""
import numpy as np

from .util import get_rng
from .exceptions import DataError, PescadorError

__all__ = ["buffer_stream", "tuples", "keras_tuples", "cache"]


def __stack_data(data, axis):
    output = dict()
    for key in data[0].keys():
        if axis is None:
            output[key] = np.array([x[key] for x in data])
        else:
            output[key] = np.concatenate([x[key] for x in data], axis=axis)

    return output


def buffer_stream(stream, buffer_size, partial=False, axis=None):
    """Buffer data from an stream into one data object.

    This is useful when a stream produces one example at a time, and you want
    to collect `buffer_size` iterates into a single object.

    Parameters
    ----------
    stream : stream
        The stream to buffer

    buffer_size : int > 0
        The number of examples to retain per batch.

    partial : bool, default=False
        If True, yield a final partial batch on under-run.

    axis : int or None
        If `None` (default), concatenate data along a new 0th axis.
        Otherwise, concatenation is performed along the specified axis.

        This is primarily useful when combining data that already has a
        dimension for buffer index, e.g., when buffering buffers.

    Yields
    ------
    batch
        A batch of size at most `buffer_size`

    Raises
    ------
    DataError
        If the stream contains items that are not data-like.

    Examples
    --------
    This example shows how to concatenate several iterates into a batch:

    >>> def mygen():
    ...     # Make items with x = number, y = parity of x
    ...     for i in range(100):
    ...         yield dict(x=np.asarray(i), y=np.asarray(i % 2))
    >>> # Make a streamer and print the first few iterates
    >>> S = pescador.Streamer(mygen)
    >>> [_ for _ in S.iterate(5)]
    [{'x': array(0), 'y': array(0)},
     {'x': array(1), 'y': array(1)},
     {'x': array(2), 'y': array(0)},
     {'x': array(3), 'y': array(1)},
     {'x': array(4), 'y': array(0)}]
    >>> # Buffer the streamer
    >>> buf = pescador.buffer_stream(S, 5)
    >>> next(buf)
    {'x': array([0, 1, 2, 3, 4]), 'y': array([0, 1, 0, 1, 0])}

    If the iterates already have a batch index dimension, we can use it
    directly.  This can be useful when the streamers already generate
    partial batches that you want to combine, rather than singletons.

    >>> def mygen_idx():
    ...     # Make items with x = number, y = parity of x
    ...     for i in range(100):
    ...         yield dict(x=np.asarray([i]), y=np.asarray([i % 2]))
    >>> # Make a streamer and print the first few iterates
    >>> S = pescador.Streamer(mygen_idx)
    >>> [_ for _ in S.iterate(5)]
    [{'x': array([0]), 'y': array([0])},
     {'x': array([1]), 'y': array([1])},
     {'x': array([2]), 'y': array([0])},
     {'x': array([3]), 'y': array([1])},
     {'x': array([4]), 'y': array([0])}]
    >>> # This is the wrong way to do it, since it will add another index
    >>> # dimension
    >>> buf_wrong = pescador.buffer_stream(S, 5)
    >>> next(buf_wrong)
    {'x': array([[0],
        [1],
        [2],
        [3],
        [4]]), 'y': array([[0],
        [1],
        [0],
        [1],
        [0]])}
    >>> # The right way to do it, using the existing buffer index
    >>> buf_right = pescador.buffer_stream(S, 5, axis=0)
    >>> next(buf_right)
    {'x': array([0, 1, 2, 3, 4]), 'y': array([0, 1, 0, 1, 0])}
    """
    data = []
    count = 0

    for item in stream:
        data.append(item)
        count += 1

        if count < buffer_size:
            continue
        try:
            yield __stack_data(data, axis=axis)
        except (TypeError, AttributeError) as ex:
            raise DataError(f"Malformed data stream: {data}") from ex
        finally:
            data = []
            count = 0

    if data and partial:
        yield __stack_data(data, axis=axis)


def tuples(stream, *keys):
    """Reformat data as tuples.

    Parameters
    ----------
    stream : iterable
        Stream of data objects.
    *keys : strings
        Keys to use for ordering data.

    Yields
    ------
    items : tuple of np.ndarrays
        Data object reformatted as a tuple.

    Raises
    ------
    DataError
        If the stream contains items that are not data-like.
    KeyError
        If a data object does not contain the requested key.
    """
    if not keys:
        raise PescadorError("Unable to generate tuples from " "an empty item set")
    for data in stream:
        try:
            yield tuple(data[key] for key in keys)
        except TypeError as ex:
            raise DataError(f"Malformed data stream: {data}") from ex


def keras_tuples(stream, inputs=None, outputs=None):
    """Reformat data objects as keras-compatible tuples.

    For more detail: https://keras.io/models/model/#fit

    Parameters
    ----------
    stream : iterable
        Stream of data objects.
    inputs : string or iterable of strings, None
        Keys to use for ordered input data.
        If not specified, returns `None` in its place.
    outputs : string or iterable of strings, default=None
        Keys to use for ordered output data.
        If not specified, returns `None` in its place.

    Yields
    ------
    x : np.ndarray, list of np.ndarray, or None
        If `inputs` is a string, `x` is a single np.ndarray.
        If `inputs` is an iterable of strings, `x` is a list of np.ndarrays.
        If `inputs` is a null type, `x` is None.
    y : np.ndarray, list of np.ndarray, or None
        If `outputs` is a string, `y` is a single np.ndarray.
        If `outputs` is an iterable of strings, `y` is a list of np.ndarrays.
        If `outputs` is a null type, `y` is None.

    Raises
    ------
    DataError
        If the stream contains items that are not data-like.
    """
    flatten_inputs, flatten_outputs = False, False
    if inputs and isinstance(inputs, str):
        inputs = [inputs]
        flatten_inputs = True
    if outputs and isinstance(outputs, str):
        outputs = [outputs]
        flatten_outputs = True

    inputs, outputs = (inputs or []), (outputs or [])
    if not inputs + outputs:
        raise PescadorError(
            "At least one key must be given for " "`inputs` or `outputs`"
        )

    for data in stream:
        try:
            x = list(data[key] for key in inputs) or None
            if len(inputs) == 1 and flatten_inputs:
                x = x[0]

            y = list(data[key] for key in outputs) or None
            if len(outputs) == 1 and flatten_outputs:
                y = y[0]

            yield (x, y)
        except TypeError as ex:
            raise DataError(f"Malformed data stream: {data}") from ex


def cache(stream, n_cache, prob=0.5, random_state=None):
    """Stochastic stream caching.

    - With probability `prob`: yield a new item from `stream` and place it in the cache
    - With probability `1-prob`: yield a previously seen item from the cache
    - When the cache exceeds size `n_cache`, a previously seen item is selected at
      random for eviction.

    Stream caching can reduce latency in producing items, particularly when the items
    are large or take a non-trivial amount of time for the underlying `stream` to produce.
    Note that the statistics of the cached stream will differ from those of `stream` because
    items in the cache may be relatively over-represented, so use with caution.

    A cached stream will generate at least as many items as the raw stream.
    Cached streams will terminate when they attempt to collect a new item
    from the input and the input has terminated.

    .. note:: The first `n_cache` items will be generated from `stream` in order.
              Caching only becomes active after this startup phase.

    Parameters
    ----------
    stream : iterable
        The stream from which to sample

    n_cache : int > 0
        The size of the cache

    prob : float in (0, 1]
        The probability with which to select a new item.
        Small values of `prob` lead to high reuse of data; `prob=1` is equivalent
        to not caching at all.

    random_state : None, int, or np.random.RandomState
        If int, random_state is the seed used by the random number generator;

        If RandomState instance, random_state is the random number generator;

        If None, the random number generator is the RandomState instance
        used by np.random.

    Yields
    ------
    data
        elements of `stream`
    """
    if n_cache <= 0:
        raise PescadorError(f"n_cache={n_cache} must be a positive integer")

    if not 0 < prob <= 1:
        raise PescadorError(
            f"prob={prob} must be a number in the range (0, 1]."
        )

    rng = get_rng(random_state)

    data_cache = []

    while True:
        if len(data_cache) < n_cache:
            # We don't have enough data yet; grab an item and yield it
            try:
                item = next(stream)
                yield item
            except StopIteration:
                break
            data_cache.append(item)

        else:
            # Otherwise, the cache is now full.
            # Pick a random index
            idx = rng.randint(low=0, high=n_cache)

            # Then flip a coin to decide whether or not to replace it
            # with a new item
            if rng.rand() <= prob:
                try:
                    data_cache[idx] = next(stream)
                except StopIteration:
                    break

            # Now yield the item
            yield data_cache[idx]
