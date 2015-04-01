#!/usr/bin/env python
'''ZMQ-baesd stream multiplexing'''

import multiprocessing as mp
import zmq
import numpy as np

try:
    import ujson as json
except ImportError:
    import json

from joblib.parallel import SafeFunction

from .util import mux

__all__ = ['zmq_mux']


def zmq_send_arrays(socket, array_payload, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""

    names = sorted(array_payload.keys())
    md = []
    payload = []

    for name in names:
        data = array_payload[name]

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        payload.append(data)

        md.append(dict(dtype=str(data.dtype),
                       shape=data.shape,
                       name=name))

    # Send the header
    p1 = json.dumps(md)

    msg = [p1] + payload

    return socket.send_multipart(msg, flags, copy=copy, track=track)


def zmq_recv_arrays(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""

    results = dict()

    msg = socket.recv_multipart(flags=flags)

    md_array = json.loads(msg[0])

    for i, md in enumerate(md_array, start=1):
        buf = buffer(msg[i])
        results[md['name']] = np.frombuffer(buf, dtype=md['dtype'])
        results[md['name']].shape = md['shape']

    return results


def __mux_worker(port, Stream=mux, **kw):

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect('tcp://localhost:{:d}'.format(port))

    try:
        # Build the stream
        for item in Stream(*kw['args'], **kw['kwargs']):
            if isinstance(item, tuple) and len(item) == 2:
                zmq_send_arrays(socket, {'X': item[0],
                                         'Y': item[1]})
            elif isinstance(item, np.ndarray):
                zmq_send_arrays(socket, {'X': item})
            else:
                raise RuntimeError('Unsupported data: ' + item)
    finally:
        # send a kill signal
        zmq_send_arrays(socket, {'exit': np.empty([])})
        context.destroy()


def zmq_mux(port, *args, **kwargs):
    '''Multiplex over zeroMQ channels.

    This is more efficient than threaded_mux because passes data by
    reference, rather than serializing.

    For now, this only works with dense datatypes (ie, ndarray), and not
    sparse matrices.


    Parameters
    ----------
    port : int > 0
        The TCP port to use

    args, kwargs
        Pass-through to `mux`

    Generates
    ---------
    X : ndarray
    X, Y : ndarray, ndarray
        The multiplexed data
    '''
    worker = mp.Process(target=SafeFunction(__mux_worker),
                        args=[port],
                        kwargs={'args': args,
                                'kwargs': kwargs})
    context = zmq.Context()

    try:
        worker.start()

        socket = context.socket(zmq.PAIR)
        socket.bind('tcp://*:{:d}'.format(port))

        # Yield from the queue as long as it's open
        finished = False
        while not finished:
            # If kill signal received, break
            # Make a poll here
            data = zmq_recv_arrays(socket)

            if 'exit' in data:
                finished = True
            elif 'Y' in data:
                yield data['X'], data['Y']
            elif 'X' in data:
                yield data['X']
            else:
                raise RuntimeError('Unknown message: ' + str(data))

    finally:
        worker.terminate()
        context.destroy()
