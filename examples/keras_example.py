# -*- coding: utf-8 -*-
"""
===============
A Keras Example
===============

An example of how to use Pescador with Keras.
"""

# Original Code source: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# author: Christopher Jacoby

from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import pescador

batch_size = 128
nb_classes = 10
num_epochs = 6

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
n_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def setup_data():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return input_shape, (X_train, Y_train), (X_test, Y_test)


def build_model(input_shape):
    model = Sequential()

    model.add(Conv2D(n_filters, kernel_size=kernel_size,
                     padding='valid',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters, kernel_size=kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def data_generator(X, y):
    '''A basic generator for the data.'''
    X = np.atleast_2d(X)
    # y's are binary vectors, and should be of shape (1, 10) after this.
    y = np.atleast_2d(y)

    n = X.shape[0]

    while True:
        i = np.random.randint(0, n)
        yield {'X': X[np.newaxis, i], 'y': y[np.newaxis, i]}


def noisy_generator(X, y, scale=1e-1):
    '''A modified version of the original generator which adds gaussian
    noise to the original sample.
    '''
    noise_shape = X.shape[1:]
    for sample in data_generator(X, y):
        noise = scale * np.random.randn(*noise_shape)

        yield {'X': sample['X'] + noise, 'y': sample['y']}


input_shape, (X_train, Y_train), (X_test, Y_test) = setup_data()
steps_per_epoch = len(X_train) // batch_size

# Create two streams from the same data, where one of the streams
# adds a small amount of gaussian noise. You could easily perform
# other data augmentations using the same basic strategy.
basic_stream = pescador.Streamer(data_generator, X_train, Y_train)
noisy_stream = pescador.Streamer(noisy_generator, X_train, Y_train)

# Mux the two streams together.
mux = pescador.mux.Mux([basic_stream, noisy_stream],
                       # Two streams, always active.
                       2,
                       # We want to sample from each stream infinitely.
                       lam=None)

# Generate batches from the stream
training_streamer = pescador.BufferedStreamer(mux, batch_size)

model = build_model(input_shape)
model.fit_generator(
    training_streamer.tuples('X', 'y', cycle=True),
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    verbose=1,
    validation_data=(X_test, Y_test),
    validation_steps=100)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
