# -*- coding: utf-8 -*-
"""
==================
ZMQ Example
==================

An example of how to use a ZMQStreamer to generate samples,
with some small benchmarks along the way.
"""

# Imports
import numpy as np
import pescador
import time

##############################################
# Batch Generator
##############################################
# As always, you have to start with a generator function, which yields
# some simple batches. Since this is a toy example, we're just
# yielding some random numbers of the appropriate shape.
#
# It is important to remember that the first dimension is always the "samples
# dimension" (batch size), since the BufferedStreamer will concatenate batch
# components together along this dimension. Therefore, we have to force the
# target to be of 2 dimensions.


def batch_gen():
    """
    Returns
    -------
    batch_dict : dict
        A batch which looks like it might come from some
        machine learning problem, with X as Features, and Y as targets.
    """
    while True:
        yield dict(X=np.random.random((1, 10)),
                   Y=np.atleast_2d(np.random.randint(10)))


##############################################
# Basic ZMQ Usage
##############################################
# Here is a trivial ZMQStreamer example, using it directly on top of a single
# Streamer. We leave it to your imagination to decide what you would actually
# do with the batches you receive here.

n_test_batches = 1e3

# Construct a streamer
s = pescador.Streamer(batch_gen)

# Wrap teh streamer in a ZMQ streamer
zs = pescador.ZMQStreamer(s)

# Get batches from the stream as you would normally.
t0 = time.time()
batch_count = 0
for batch in zs(max_batches=n_test_batches):
    batch_count += len(batch['X'])
    # Train your network, etc.


duration = time.time() - t0
print("Generated {} samples from ZMQ\n\t"
      "in {:.5f}s ({:.5f} / sample)".format(
          batch_count, duration, duration / batch_count))

# Outputs:
# > Generated 1000 samples from ZMQ
# >   in 0.57073s (0.00057 / sample)

##############################################
# Buffering ZMQ
##############################################
# You could also wrap the ZMQStreamer in a BufferedStreamer, to produce
# "mini-batches" for training, etc.
#
# Note: You could put the BufferedStreamer before or after the ZMQStreamer;
# it sould work both ways.
buffer_size = 10
buffered_zmq = pescador.BufferedStreamer(zs, buffer_size)

# Get batches from the stream as you would normally.
iter_count = 0
batch_count = 0
t0 = time.time()
for batch in buffered_zmq(max_batches=n_test_batches):
    iter_count += 1
    batch_count += len(batch['X'])

duration = time.time() - t0
print("Generated {} batches of {} samples from Buffered ZMQ Streamer"
      "\n\tin {:.5f}s ({:.5f} / sample)".format(iter_count,
                                                batch_count,
                                                duration,
                                                duration / batch_count))

# Outputs
# > Generated 1000 batches of 10000 samples from Buffered ZMQ Streamer
# >     in 1.69138s (0.00017 / sample)
