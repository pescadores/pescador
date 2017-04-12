#!/usr/bin/env python
'''Map functions'''
import numpy as np
import six

from .exceptions import PescadorError
from . import util

__all__ = ['buffer_stream']


def __stack_data(data):
    output = dict()
    for key in data[0].keys():
        output[key] = np.array([x[key] for x in data])
    return output


def buffer_stream(stream, buffer_size, partial=False,
                  generator=util.Deprecated()):
    '''Buffer "data" from an stream into one data object.

    Parameters
    ----------
    stream : stream
        The stream to buffer

    buffer_size : int > 0
        The number of examples to retain per batch.

    partial : bool, default=False
        If True, yield a final partial batch on under-run.

    generator : stream
        .. warning:: This parameter name was deprecated in pescador 1.1.0
            Use the `stream` parameter instead.
            The `generator` parameter will be removed in pescador 2.0.0.
    Yields
    ------
    batch
        A batch of size at most `buffer_size`
    '''

    stream = util.rename_kw('generator', generator,
                            'stream', stream,
                            '1.1.0', '2.0.0')

    data = []
    n = 0

    for x in stream:
        data.append(x)
        n += 1

        if n < buffer_size:
            continue
        try:
            yield __stack_data(data)
        except TypeError:
            raise PescadorError("Malformed data stream: {}".format(data))
        finally:
            data = []
            n = 0

    if data and partial:
        yield __stack_data(data)


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
    """
    if inputs and isinstance(inputs, six.string_types):
        inputs = [inputs]
    if outputs and isinstance(outputs, six.string_types):
        outputs = [outputs]

    inputs, outputs = (inputs or []), (outputs or [])
    if not inputs + outputs:
        raise PescadorError('At least one key must be given for '
                            '`inputs` or `outputs`')

    for data in stream:
        try:
            x = list(data[key] for key in inputs) or None
            if len(inputs) == 1:
                x = x[0]

            y = list(data[key] for key in outputs) or None
            if len(outputs) == 1:
                y = y[0]

            yield (x, y)
        except TypeError:
            raise PescadorError("Malformed data stream: {}".format(data))
