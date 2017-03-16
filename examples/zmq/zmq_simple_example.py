"""
==================
Simple ZMQ Example
==================
A trivial ZMQ example.
"""

import numpy as np
import pescador


def npy_generator(filepath):
    pass


# Construct a streamer
s = pescador.streamer(npy_generator)

# Wrap teh streamer in a ZMQ streamer
zs = pescador.ZMQStreamer(s)

# Get batches from the stream as you would normally.
batch_count = 0
for batch in zs.generate(max_batches=10):
    batch_count += len(batch)

    # Train your network, etc.

print("")
