# -*- coding: utf-8 -*-
"""
=================
Mux Files Example
=================

Let's say you have a machine learning task like digit recognition.
You have multiple datasets, and you would like to sample from them evenly.
Pescador's Mux Streamer is a perfect tool to facilitate this sort of setup.

For this example, to simulate this experience, we will split the canonical
MNIST training set evenly into three pieces, and save them to thier own
.npy files.
"""

from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import ShuffleSplit

import pescador

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

##############################################
# Prepare Datasets
##############################################
dataset1_path = "/tmp/dataset1_train.npz"
dataset2_path = "/tmp/dataset2_train.npz"
dataset3_path = "/tmp/dataset3_train.npz"
datasets = [dataset1_path, dataset2_path, dataset3_path]


def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return input_shape, (x_train, y_train), (x_test, y_test)


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


input_shape, (X_train, Y_train), (X_test, Y_test) = load_mnist()
split_and_save_datasets(X_train, Y_train, datasets)

##############################################
# Create Generator and Streams for each dataset.
##############################################


def npz_generator(npz_path):
    """Generate data from an npz file."""
    npz_data = np.load(npz_path)
    X = npz_data['X']
    # y's are binary vectors, and should be of shape (1, 10) after this.
    y = npz_data['Y']

    n = X.shape[0]

    while True:
        i = np.random.randint(0, n)
        yield {'X': X[np.newaxis, i], 'Y': y[np.newaxis, i]}


streams = [pescador.Streamer(npz_generator, x) for x in datasets]


##############################################
# Option 1: Stream equally from each dataset
##############################################
# If you can easily fit all the datasets in memory and you want to
# sample from then equally, you would set up your Mux as follows:

mux = pescador.mux.Mux(streams,
                       # Three streams, always active.
                       k=len(streams),
                       # We want to sample from each stream infinitely,
                       # so we turn off the lam parameter, which
                       # controlls how long to sample from each stream.
                       lam=None)


##############################################
# Option 2: Sample from one at a time.
##############################################
# Another approach might be to restrict sampling to one stream at a time.
# Now, the lam parameter controlls (statistically) how long to sapmle
# from a stream before activating a new stream.

mux = pescador.mux.Mux(streams,
                       # Only allow one active stream
                       k=1,
                       # Sample on average 1000 samples from a stream before
                       # moving onto another one.
                       lam=1000)


##############################################
# Use the mux as a streamer
##############################################
# At this point, you can use the Mux as a streamer normally.
