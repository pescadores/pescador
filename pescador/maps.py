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
from . import util

__all__ = ['buffer_stream', 'tuples', 'keras_tuples']


def __stack_data(data):
    output = dict()
    for key in data[0].keys():
        output[key] = np.array([x[key] for x in data])
    return output


def buffer_stream(stream, buffer_size, partial=False):
    '''Buffer "data" from an stream into one data object.

    Parameters
    ----------
    stream : stream
        The stream to buffer

    buffer_size : int > 0
        The number of examples to retain per batch.

    partial : bool, default=False
        If True, yield a final partial batch on under-run.

    Yields
    ------
    batch
        A batch of size at most `buffer_size`

    Raises
    ------
    DataError
        If the stream contains items that are not data-like.
    '''

    data = []
    n = 0

    for x in stream:
        data.append(x)
        n += 1

        if n < buffer_size:
            continue
        try:
            yield __stack_data(data)
        except (TypeError, AttributeError):
            raise DataError("Malformed data stream: {}".format(data))
        finally:
            data = []
            n = 0

    if data and partial:
        yield __stack_data(data)


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
