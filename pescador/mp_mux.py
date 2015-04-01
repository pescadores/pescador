#!/usr/bin/env python
'''Multiprocessing-based stream multiplexing'''

import multiprocessing as mp
import ctypes
import Queue
from joblib.parallel import SafeFunction

from .util import mux


def threaded_mux(q_size, *args, **kwargs):
    '''A threaded version of stream multiplexor.

    :parameters:
        - q_size : int >= 0
          If positive-valued, the maximum number of items to allow in the
          inter-process queue.  Otherwise, the queue is unboundedly large.

          Probably 1000 is a good value for most scenarios.

        - args, kwargs
          See: mux()
    '''

    def __mux_worker(data_queue, done, exc_queue, **kw):
        '''Wrapper function to iterate a mux stream and queue the results'''

        try:
            # Build the stream
            mux_stream = mux(*kw['args'], **kw['kwargs'])

            # Push into the queue, blocking if it's full
            for item in mux_stream:
                data_queue.put(item)

        except Exception as exc:
            exc_queue.put(exc)

        finally:
            # Cleanup actions
            # close the queue

            with done.get_lock():
                # Signal that we're done
                done.value = True
                data_queue.close()
                exc_queue.close()

    # Construct a queue object
    data_queue = mp.Queue(maxsize=q_size)
    exc_queue = mp.Queue(maxsize=1)
    done = mp.Value(ctypes.c_bool)

    worker = mp.Process(target=SafeFunction(__mux_worker),
                        args=[data_queue, done, exc_queue],
                        kwargs={'args': args, 'kwargs': kwargs})

    worker.start()

    # Yield from the queue as long as it's open
    while True:
        with done.get_lock():
            my_done = done.value

        if my_done and data_queue.empty():
            break

        try:
            yield data_queue.get(block=False, timeout=1e-2)
        except Queue.Empty:
            pass

        try:
            exc = exc_queue.get_nowait()
            raise exc
        except Queue.Empty:
            pass

    exc_queue.close()
    exc_queue.join_thread()
