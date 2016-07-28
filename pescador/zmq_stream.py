#!/usr/bin/env python
'''ZMQ-baesd stream multiplexing

.. autosummary::
    :toctree: generated/

    zmq_stream
'''

import multiprocessing as mp
import zmq
import numpy as np
import six
import sys
import warnings

try:
    import ujson as json
except ImportError:
    import json

try:
    # joblib <= 0.9.4
    from joblib.parallel import SafeFunction
except ImportError:
    # joblib >= 0.10.0
    from joblib._parallel_backends import SafeFunction

__all__ = ['zmq_stream']


# A hack to support buffers in py3
if six.PY3:
    buffer = memoryview


def zmq_send_batch(socket, batch, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""

    header, payload = [], []

    for key in sorted(batch.keys()):
        data = batch[key]

        if not isinstance(data, np.ndarray):
            raise TypeError('Only ndarray types can be serialized')

        header.append(dict(dtype=str(data.dtype),
                           shape=data.shape,
                           key=key,
                           aligned=data.flags['ALIGNED']))
        # Force contiguity
        payload.append(data)

    # Send the header
    msg = [json.dumps(header).encode('ascii')]
    msg.extend(payload)

    return socket.send_multipart(msg, flags, copy=copy, track=track)


def zmq_recv_batch(socket, flags=0, copy=True, track=False):
    """recv a batch"""

    results = dict()

    msg = socket.recv_multipart(flags=flags, copy=copy, track=track)

    headers = json.loads(msg[0].decode('ascii'))

    if len(headers) == 0:
        raise StopIteration

    for header, payload in zip(headers, msg[1:]):
        results[header['key']] = np.frombuffer(buffer(payload),
                                               dtype=header['dtype'])
        results[header['key']].shape = header['shape']
        if six.PY2:
            # Legacy python won't let us preserve alignment, skip this step
            continue
        results[header['key']].flags['ALIGNED'] = header['aligned']

    return results


def zmq_worker(port, streamer, terminate, copy=False, max_batches=None):

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect('tcp://localhost:{:d}'.format(port))

    try:
        # Build the stream
        for batch in streamer.generate(max_batches=max_batches):
            if terminate.is_set():
                break
            zmq_send_batch(socket, batch, copy=copy)

    finally:
        # send an empty payload to kill
        zmq_send_batch(socket, {})
        context.destroy()


def zmq_stream(streamer, max_batches=None,
               min_port=49152, max_port=65535, max_tries=100,
               copy=False, timeout=None):
    '''Parallel data streaming over zeromq sockets.

    This allows a data generator to run in a separate process
    from the consumer.

    A typical usage pattern is to construct a `Streamer` object
    from a generator (or `util.mux` of several `Streamer`s),
    and then use `zmq_stream` to execute the stream in one process
    while the other process consumes data, e.g., with a `StreamLearner`
    object.

    Parameters
    ----------
    streamer : `pescador.Streamer`
        The streamer object

    max_batches : None or int > 0
        Maximum number of batches to generate

    min_port : int > 0
    max_port : int > min_port
        The range of TCP ports to use

    max_tries : int > 0
        The maximum number of connection attempts to make

    copy : bool
        Set `True` to enable data copying

    Yields
    ------
    batch
        Data drawn from `streamer.generate(max_batches)`.
    '''
    context = zmq.Context()

    if six.PY2:
        warnings.warn('zmq_stream cannot preserve numpy array alignment in Python 2',
                      RuntimeWarning)

    try:
        socket = context.socket(zmq.PAIR)

        port = socket.bind_to_random_port('tcp://*',
                                          min_port=min_port,
                                          max_port=max_port,
                                          max_tries=max_tries)
        terminate = mp.Event()

        worker = mp.Process(target=SafeFunction(zmq_worker),
                            args=[port, streamer, terminate],
                            kwargs=dict(copy=copy,
                                        max_batches=max_batches))

        worker.daemon = True
        worker.start()

        # Yield from the queue as long as it's open
        while worker.is_alive():
            yield zmq_recv_batch(socket)

    except StopIteration:
        pass

    except:
        six.reraise(*sys.exc_info())

    finally:
        terminate.set()
        worker.join(timeout)
        if worker.is_alive():
            worker.terminate()
        context.destroy()
