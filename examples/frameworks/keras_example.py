# -*- coding: utf-8 -*-
"""
===============
A Keras Example
===============

An example of how to use Pescador with Keras.

3/18/2017: Updated to Keras 2.0 API.
"""

# Original Code source: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

##############################################
# Setup and Definitions
##############################################

from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

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

##############################################
# Setup Keras model
##############################################


def build_model(input_shape):
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
# Define Data Generators
##############################################
# To add a little bit of complexity, and show a little of what you could
# do with Keras, we'll add an additional generator which simply
# adds a little gaussian noise to the data.


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


##############################################
# Put it all together
##############################################
# They key method for interfacing with Keras is the `Streamer.tuples()`,
# function of the streamer, which takes args of the batch key names to pass to
# Keras, since Keras's `fit_generator` consumes tuples from the generator.

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
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
