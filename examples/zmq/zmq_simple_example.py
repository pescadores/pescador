"""
==================
Simple ZMQ Example
==================
A trivial ZMQ example.
"""

import numpy as np
import pescador


def batch_gen():
    """
    Returns
    -------
    batch_dict
        A batch which looks like it might come from some
        machine learning problem, with X as Features, and Y as targets.
    """
    yield dict(X=np.random.random((1, 10)),
               Y=np.random.randint(10))


# Construct a streamer
s = pescador.Streamer(batch_gen)

# Wrap teh streamer in a ZMQ streamer
zs = pescador.ZMQStreamer(s)

# Get batches from the stream as you would normally.
batch_count = 0
for batch in zs.generate(max_batches=10):
    batch_count += len(batch)

    # Train your network, etc.

print("Generated {} batches from ZMQ".format(batch_count))

# Now, you could also wrap the ZMQStreamer in a BufferedStreamer, like so:
buffer_size = 10
buffered_zmq = pescador.BufferedStreamer(zs, buffer_size)

# Get batches from the stream as you would normally.
batch_count = 0
for batch in zs.generate(max_batches=10):
    batch_count += len(batch)

    print("Received {} batches.")

print("Generated {} batches from Buffered ZMQ Streamer".format(batch_count))
