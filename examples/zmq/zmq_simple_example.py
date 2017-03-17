"""
==================
Simple ZMQ Example
==================

An example of how to use a ZMQStreamer to generate samples,
with some small benchmarks along the way.
"""

import numpy as np
import pescador
import time


def batch_gen():
    """
    Returns
    -------
    batch_dict
        A batch which looks like it might come from some
        machine learning problem, with X as Features, and Y as targets.
    """
    while True:
        # For both of these, the first dimension is the number of samples.
        # Therefore, we have to force the target to be of 2 dimensions.
        yield dict(X=np.random.random((1, 10)),
                   Y=np.atleast_2d(np.random.randint(10)))


n_test_batches = 1e3

##############################################
# Basic ZMQ Usage
##############################################

# Construct a streamer
s = pescador.Streamer(batch_gen)

# Wrap teh streamer in a ZMQ streamer
zs = pescador.ZMQStreamer(s)

# Get batches from the stream as you would normally.
t0 = time.time()
batch_count = 0
for batch in zs.generate(max_batches=n_test_batches):
    batch_count += len(batch['X'])

    # Train your network, etc.
duration = time.time() - t0
print("Generated {} samples from ZMQ\n\t"
      "in {:.5f}s ({:.5f} / sample)".format(
          batch_count, duration, duration / batch_count))

##############################################
# Buffering ZMQ
##############################################
# Now, you could also wrap the ZMQStreamer in a BufferedStreamer, like so:
buffer_size = 10
buffered_zmq = pescador.BufferedStreamer(zs, buffer_size)

# Get batches from the stream as you would normally.
iter_count = 0
batch_count = 0
t0 = time.time()
for batch in buffered_zmq.generate(max_batches=n_test_batches):
    iter_count += 1
    batch_count += len(batch['X'])

duration = time.time() - t0
print("Generated {} batches of {} samples from Buffered ZMQ Streamer"
      "\n\tin {:.5f}s ({:.5f} / sample)".format(iter_count,
                                                batch_count,
                                                duration,
                                                duration / batch_count))
