#!/usr/bin/env python
"""
Parallel streaming
------------------

Streaming from background processes is implemented with the ZeroMQ library.
The `ZMQStreamer` object wraps ordinary streamers (or muxes) for background execution.

.. autosummary::
    :toctree: generated/

    ZMQStreamer

"""

import multiprocessing as mp
import zmq
import numpy as np
import msgpack

from .core import Streamer
from .exceptions import DataError


__all__ = ["ZMQStreamer"]


def zmq_send_data(socket, data, flags=0, copy=True, track=False):
    """Send data, e.g. {key: np.ndarray}, with metadata"""
    header, payload = [], []

    for key in sorted(data.keys()):
        arr = data[key]

        if not isinstance(arr, np.ndarray):
            raise DataError("Only ndarray types can be serialized")

        header.append(
            dict(
                dtype=str(arr.dtype),
                shape=arr.shape,
                key=key,
                aligned=arr.flags["ALIGNED"],
            )
        )
        # Force contiguity
        payload.append(arr)

    # Send the header
    msg = [msgpack.packb(header)]
    msg.extend(payload)

    return socket.send_multipart(msg, flags, copy=copy, track=track)


def zmq_recv_data(socket, flags=0, copy=True, track=False):
    """Receive data over a socket."""
    data = dict()

    msg = socket.recv_multipart(flags=flags, copy=copy, track=track)

    headers = msgpack.unpackb(msg[0], raw=False)

    if len(headers) == 0:
        raise StopIteration

    for header, payload in zip(headers, msg[1:]):
        data[header["key"]] = np.frombuffer(memoryview(payload), dtype=header["dtype"])
        data[header["key"]].shape = header["shape"]
        data[header["key"]].flags["ALIGNED"] = header["aligned"]

    return data


def zmq_worker(port, streamer, terminate, copy=False, max_iter=None):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    # TODO: Open this up to support different hosts.
    socket.connect(f"tcp://localhost:{port:d}")

    try:
        # Build the stream
        for data in streamer(max_iter=max_iter):
            zmq_send_data(socket, data, copy=copy)
            if terminate.is_set():
                break

    finally:
        # send an empty payload to kill
        zmq_send_data(socket, {})
        context.destroy()


class ZMQStreamer(Streamer):
    """Parallel data streaming over zeromq sockets.

    This allows a data generator to run in a separate process
    from the consumer.

    A typical usage pattern is to construct a `Streamer` object
    from a generator and then use `ZMQStreamer` to execute the stream in one
    process while the other process consumes data.


    Examples
    --------
    >>> # Construct a streamer object
    >>> S = pescador.Streamer(my_generator)
    >>> # Wrap the streamer in a ZMQ streamer
    >>> Z = pescador.ZMQStreamer(S)
    >>> # Process as normal
    >>> for data in Z:
    ...     MY_FUNCTION(data)
    """

    def __init__(
        self,
        streamer,
        min_port=49152,
        max_port=65535,
        max_tries=100,
        copy=False,
        timeout=5,
    ):
        """
        Parameters
        ----------
        streamer : `pescador.Streamer`
            The streamer object
        min_port : int > 0
        max_port : int > min_port
            The range of TCP ports to use
        max_tries : int > 0
            The maximum number of connection attempts to make
        copy : bool
            Set `True` to enable data copying
        timeout : [optional] number > 0
            Maximum time (in seconds) to wait before killing subprocesses.
            If `None`, then the streamer will wait indefinitely for
            subprocesses to terminate.
        """
        self.streamer = streamer
        self.min_port = min_port
        self.max_port = max_port
        self.max_tries = max_tries
        self.copy = copy
        self.timeout = timeout

    def iterate(self, max_iter=None):
        """
        Note: A ZMQStreamer does not activate its stream,
        but allows the zmq_worker to do that.

        Yields
        ------
        data : dict
            Data drawn from `streamer(max_iter)`.
        """
        context = zmq.Context()

        try:
            socket = context.socket(zmq.PAIR)

            port = socket.bind_to_random_port(
                "tcp://*",
                min_port=self.min_port,
                max_port=self.max_port,
                max_tries=self.max_tries,
            )
            terminate = mp.Event()

            worker = mp.Process(
                target=zmq_worker,
                args=[port, self.streamer, terminate],
                kwargs=dict(copy=self.copy, max_iter=max_iter),
            )

            worker.daemon = True
            worker.start()

            # Yield from the queue as long as it's open
            while True:
                yield zmq_recv_data(socket)

        except StopIteration:
            pass

        finally:
            terminate.set()
            if worker.is_alive():
                worker.join(self.timeout)
            if worker.is_alive():
                worker.terminate()
            context.destroy()
