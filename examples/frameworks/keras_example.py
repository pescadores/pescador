# -*- coding: utf-8 -*-
"""
===============
A Keras Example
===============

An example of how to use Pescador with Keras.

Original Code source:
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
"""

##############################################
# Setup and Definitions
##############################################

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

import pescador

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28


##############################################
# Load and preprocess data
##############################################

def setup_data():
    """Load and shape data for training with Keras + Pescador.

    Returns
    -------
    input_shape : tuple, len=3
        Shape of each sample; adapts to channel configuration of Keras.

    X_train, y_train : np.ndarrays
        Images and labels for training.

    X_test, y_test : np.ndarrays
        Images and labels for test.
    """
    # The data, shuffled and split between train and test sets
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


##############################################
# Setup Keras model
##############################################

def build_model(input_shape):
    """Create a compiled Keras model.

    Parameters
    ----------
    input_shape : tuple, len=3
        Shape of each image sample.

    Returns
    -------
    model : keras.Model
        Constructed model.
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


##############################################
# Define Data Sampler
##############################################

def sampler(X, y):
    '''A basic generator for sampling data.

    Parameters
    ----------
    X : np.ndarray, len=n_samples, ndim=4
        Image data.

    y : np.ndarray, len=n_samples, ndim=2
        One-hot encoded class vectors.

    Yields
    ------
    data : dict
        Single image sample, like {X: np.ndarray, y: np.ndarray}
    '''
    X = np.atleast_2d(X)
    # y's are binary vectors, and should be of shape (10,) after this.
    y = np.atleast_1d(y)

    n = X.shape[0]

    while True:
        i = np.random.randint(0, n)
        yield {'X': X[i], 'y': y[i]}


##############################################
# Define a Custom Map Function
##############################################

def additive_noise(stream, key='X', scale=1e-1):
    '''Add noise to a data stream.

    Parameters
    ----------
    stream : iterable
        A stream that yields data objects.

    key : string, default='X'
        Name of the field to add noise.

    scale : float, default=0.1
        Scale factor for gaussian noise.

    Yields
    ------
    data : dict
        Updated data objects in the stream.
    '''
    for data in stream:
        noise_shape = data[key].shape
        noise = scale * np.random.randn(*noise_shape)
        data[key] = data[key] + noise
        yield data


##############################################
# Put it all together
##############################################
input_shape, (X_train, Y_train), (X_test, Y_test) = setup_data()
steps_per_epoch = len(X_train) // batch_size

# Create two streams from the same data, where one of the streams
# adds a small amount of Gaussian noise. You could easily perform
# other data augmentations using the same 'map' strategy.
stream = pescador.Streamer(sampler, X_train, Y_train)
noisy_stream = pescador.Streamer(additive_noise, stream, 'X')

# Multiplex the two streamers together.
mux = pescador.StochasticMux([stream, noisy_stream],
                             # Two streams, always active.
                             n_active=2,
                             # We want to sample from each stream infinitely.
                             rate=None)

# Buffer the stream into minibatches.
batches = pescador.buffer_stream(mux, batch_size)

model = build_model(input_shape)
try:
    print("Start time: {}".format(datetime.datetime.now()))
    model.fit_generator(
        pescador.tuples(batches, 'X', 'y'),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test))
except KeyboardInterrupt:
    print("Stopping early")
finally:
    print("Finished: {}".format(datetime.datetime.now()))
    scores = model.evaluate(X_test, Y_test, verbose=0)
    for val, name in zip(scores, model.metrics_names):
        print(f'Test {name}: {val:0.4f}')
