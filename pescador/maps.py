#!/usr/bin/env python
'''Map functions perform operations on a stream.

Important note: map functions return a *generator*, not another
Streamer, so if you need it to behave like a Streamer, you have to wrap
the function in a Streamer again.

.. autosummary::
    :toctree: generated/

    buffer_stream
    tuples
    keras_tuples
'''
import numpy as np
import six

from .exceptions import DataError, PescadorError

__all__ = ['buffer_stream', 'tuples', 'keras_tuples']


def __stack_data(data, axis):
    output = dict()
    for key in data[0].keys():
        if axis is None:
            output[key] = np.array([x[key] for x in data])
        else:
            output[key] = np.concatenate([x[key] for x in data], axis=axis)

    return output


def buffer_stream(stream, buffer_size, partial=False, axis=None):
    '''Buffer data from an stream into one data object.

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
    '''

    data = []
    count = 0

    for item in stream:
        data.append(item)
        count += 1

        if count < buffer_size:
            continue
        try:
            yield __stack_data(data, axis=axis)
        except (TypeError, AttributeError):
            raise DataError("Malformed data stream: {}".format(data))
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
        Data object reformated as a tuple.

    Raises
    ------
    DataError
        If the stream contains items that are not data-like.
    KeyError
        If a data object does not contain the requested key.
    """
    if not keys:
        raise PescadorError('Unable to generate tuples from '
                            'an empty item set')
    for data in stream:
        try:
            yield tuple(data[key] for key in keys)
        except TypeError:
            raise DataError("Malformed data stream: {}".format(data))


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
    if inputs and isinstance(inputs, six.string_types):
        inputs = [inputs]
        flatten_inputs = True
    if outputs and isinstance(outputs, six.string_types):
        outputs = [outputs]
        flatten_outputs = True

    inputs, outputs = (inputs or []), (outputs or [])
    if not inputs + outputs:
        raise PescadorError('At least one key must be given for '
                            '`inputs` or `outputs`')

    for data in stream:
        try:
            x = list(data[key] for key in inputs) or None
            if len(inputs) == 1 and flatten_inputs:
                x = x[0]

            y = list(data[key] for key in outputs) or None
            if len(outputs) == 1 and flatten_outputs:
                y = y[0]

            yield (x, y)
        except TypeError:
            raise DataError("Malformed data stream: {}".format(data))
