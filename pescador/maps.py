#!/usr/bin/env python
'''Map functions'''
import numpy as np

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
