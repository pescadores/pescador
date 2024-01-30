# -*- coding: utf-8 -*-
"""
=================================
Muxing Multiple Datasets Together
=================================

Let's say you have a machine learning task like digit recognition.
You have multiple datasets, and you would like to sample from them evenly.
Pescador's Mux Streamer is a perfect tool to facilitate this sort of setup.

For this example, to simulate this experience, we will split the canonical
MNIST training set evenly into three pieces, and save them to their own
.npy files.
"""

import numpy as np
from keras.datasets import mnist

import pescador


##############################################
# Prepare Datasets
##############################################
dataset1_path = "dataset1_train.npz"
dataset2_path = "dataset2_train.npz"
dataset3_path = "dataset3_train.npz"
datasets = [dataset1_path, dataset2_path, dataset3_path]


def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return (x_train, y_train), (x_test, y_test)


def split_and_save_datasets(X, Y, paths):
    """Shuffle X and Y into n / len(paths) datasets, and save them
    to disk at the locations provided in paths.
    """
    shuffled_idxs = np.random.permutation(np.arange(len(X)))

    for i in range(len(paths)):
        # Take every len(paths) item, starting at i.
        # len(paths) is 3, so this would be [0::3], [1::3], [2::3]
        X_i = X[shuffled_idxs[i::len(paths)]]
        Y_i = Y[shuffled_idxs[i::len(paths)]]
        np.savez(paths[i], X=X_i, Y=Y_i)


(X_train, Y_train), (X_test, Y_test) = load_mnist()
split_and_save_datasets(X_train, Y_train, datasets)

#################################################
# Create Generator and Streams for each dataset.
#################################################


@pescador.streamable
def npz_generator(npz_path):
    """Generate data from an npz file."""
    npz_data = np.load(npz_path)
    X = npz_data['X']
    # Y is a binary maxtrix with shape=(n, k), each y will have shape=(k,)
    y = npz_data['Y']

    n = X.shape[0]

    while True:
        i = np.random.randint(0, n)
        yield {'X': X[i], 'Y': y[i]}


streams = [npz_generator(x) for x in datasets]


##############################################
# Option 1: Stream equally from each dataset
##############################################
# If you can easily fit all the datasets in memory and you want to
# sample from then equally, you would set up your Mux as follows:

mux = pescador.StochasticMux(streams,
                             # Three streams, always active.
                             n_active=len(streams),
                             # We want to sample from each stream infinitely,
                             # so we turn off the rate parameter, which
                             # controls how long to sample from each stream.
                             rate=None)


##############################################
# Option 2: Sample from one at a time.
##############################################
# Another approach might be to restrict sampling to one stream at a time.
# Now, the rate parameter controls (statistically) how long to sample
# from a stream before activating a new stream.

mux = pescador.StochasticMux(streams,
                             # Only allow one active stream
                             n_active=1,
                             # Sample on average 1000 samples from a stream before
                             # moving onto another one.
                             rate=1000)


##############################################
# Use the mux as a streamer
##############################################
# At this point, you can use the Mux as a streamer normally:

for data in mux(max_iter=10):
    print({k: v.shape for k, v in data.items()})
