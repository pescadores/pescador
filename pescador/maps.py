#!/usr/bin/env python
'''Map functions'''
import numpy as np

from .exceptions import PescadorError


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
        except TypeError:
            raise PescadorError("Malformed data stream: {}".format(data))
        finally:
            data = []
            n = 0

    if data and partial:
        yield __stack_data(data)
