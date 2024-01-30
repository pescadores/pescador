"""
==================
Parallel streaming
==================

An example of how to use a ZMQStreamer to generate samples
in a background process, with some small benchmarks along the way.
"""

# Imports
import numpy as np
import pescador
import time


##############################################
# Computational Load
##############################################
# Often, the act of generating samples will be computationally
# expensive, place a heavy load on disk I/O, or both. Here, we
# can mimic costly processing by doing a bunch of pointless math
# on an array.

def costly_function(X, n_ops=100):
    for n in range(n_ops):
        if (n % 2):
            X = X ** 3.0
        else:
            X = X ** (1. / 3)
    return X


##############################################
# Sample Generator
##############################################
# Here we define a sampler function that yields
# some simple data. We'll run some computation on the inside to
# slow things down a bit.

@pescador.streamable
def data_gen(n_ops=100):
    """Yield data, while optionally burning compute cycles.

    Parameters
    ----------
    n_ops : int, default=100
        Number of operations to run between yielding data.

    Returns
    -------
    data : dict
        A object which looks like it might come from some
        machine learning problem, with X as features, and y as targets.
    """
    while True:
        X = np.random.uniform(size=(64, 64))
        yield dict(X=costly_function(X, n_ops),
                   y=np.random.randint(10, size=(1,)))


def timed_sampling(stream, n_iter, desc):
    start_time = time.time()
    for data in stream(max_iter=n_iter):
        costly_function(data['X'])

    duration = time.time() - start_time
    print("{} :: Average time per iteration: {:0.3f} sec"
          .format(desc, duration / max_iter))


max_iter = 1e2

# Construct a streamer
timed_sampling(data_gen, max_iter, 'Single-threaded')
# Single-threaded :: Average time per iteration: 0.024 sec


##############################################
# Basic ZMQ Usage
##############################################
# Here is a trivial ZMQStreamer example, using it directly on top of a single
# Streamer. We leave it to your imagination to decide what you would actually
# do with the batches you receive here.

# Wrap the streamer in a ZMQ streamer
zstream = pescador.ZMQStreamer(data_gen)
timed_sampling(zstream, max_iter, 'ZMQ')
# ZMQ :: Average time per iteration: 0.012 sec


##############################################
# ZMQ with Map Functions
##############################################
# You will also likely want to buffer samples for building mini-batches. Here,
# we demonstrate best practices for using map functions in a stream pipeline.

buffer_size = 16

# Get batches from the stream as you would normally.
batches = pescador.Streamer(pescador.buffer_stream, data_gen, buffer_size)
timed_sampling(batches, max_iter, 'Single-threaded Batches')
# Single-threaded Batches :: Average time per iteration: 0.392 sec


zstream = pescador.ZMQStreamer(batches)
timed_sampling(zstream, max_iter, 'ZMQ Batches')
# ZMQ Batches :: Average time per iteration: 0.196 sec
