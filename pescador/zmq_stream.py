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

__all__ = ['zmq_stream']


def zmq_send_batch(socket, batch, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""

    header, payload = [], []

    for key in sorted(batch.keys()):
        data = batch[key]

        if not isinstance(data, np.ndarray):
            raise TypeError('Only ndarray types can be serialized')

        header.append(dict(dtype=str(data.dtype),
                           shape=data.shape,
                           key=key))
        payload.append(data)

    # Send the header
    msg = [json.dumps(header)]
    msg.extend(payload)

    return socket.send_multipart(msg, flags, copy=copy, track=track)


def zmq_recv_arrays(socket, flags=0, copy=True, track=False):
    """recv a batch"""

    results = dict()

    msg = socket.recv_multipart(flags=flags, copy=copy, track=track)

    headers = json.loads(msg[0])

    if len(headers) == 0:
        raise StopIteration

    for header, payload in zip(headers, msg[1:]):
        results[header['key']] = np.frombuffer(buffer(payload),
                                               dtype=header['dtype'])
        results[header['key']].shape = header['shape']

    return results


def zmq_worker(port, streamer, max_items=None):

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect('tcp://localhost:{:d}'.format(port))

    try:
        # Build the stream
        for batch in streamer.generate(max_items=max_items):
            zmq_send_batch(socket, batch)
    finally:
        # send an empty payload to kill
        zmq_send_batch(socket, {})
        context.destroy()


def zmq_stream(port, streamer, max_items=None):
    '''Stream over zeroMQ channels.

    This is more efficient than threaded_mux because passes data by
    reference, rather than serializing.

    For now, this only works with dense datatypes (ie, ndarray), and not
    sparse matrices.


    Parameters
    ----------
    port : int > 0
        The TCP port to use

    streamer : Streamer
        The streamer object

    max_items : None or int > 0
        Maximum number of batches to generate

    Generates
    ---------
    batch
    '''
    worker = mp.Process(target=SafeFunction(zmq_worker),
                        args=[port, streamer],
                        kwargs=dict(max_items=max_items))

    context = zmq.Context()

    try:
        worker.start()

        socket = context.socket(zmq.PAIR)
        socket.bind('tcp://*:{:d}'.format(port))

        # Yield from the queue as long as it's open
        while worker.is_alive():
            yield zmq_recv_arrays(socket)

    except StopIteration:
        pass

    finally:
        worker.terminate()
        context.destroy()
