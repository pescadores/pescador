#! -*- coding: utf-8 -*-
"""
=====================================
Using ChainMux for repeatable streams
=====================================

Some applications call for deterministic, repeatable data streams, rather
than stochastic samples.
A common use case is *validation* in machine learning applications, where
a held-out set of data is used to estimate the quality of a model during
training.
The validation score is computed repeatedly as the model changes, and the
resulting scores are compared to each other to find the best version of
the model.
The simplest way to ensure that the validation scores are comparable is to
use the same sample set each time.
With Pescador, this is most easily achieved by using the `ChainMux`.
"""

# Imports
import numpy as np
import pescador

##########################
# Setup
##########################
# We'll assume that the validation data lives in some N files
# Each file produces M examples, so the total validation set
# has N*M examples

val_files = ['file1.npz', 'file2.npz']
N = len(val_files)

M = 10  # or whatever the number of examples per file is


############################
# Data generator
############################
# We'll make a simple generator that streams the first m
# examples from an npz file.
# The npz file is assumed to store two arrays: X and Y
# containing inputs and outputs (eg, images and class labels)
# Once the streamer produces m examples, it exits.

@pescador.streamable
def data_gen(filename, m):

    data = np.load(filename)
    X = data['X']
    Y = data['Y']
    for i in range(m):
        yield dict(X=X[i], y=Y[i])


############################
# Constructing the streamers
############################
# First, we'll make a streamer for each validation example.
#

val_streams = [data_gen(fn, M) for fn in val_files]


############################
# Building the mux
############################
# The `ChainMux` can be used to combine data from all val_streams
# in order.
# We'll use `cycle` mode here, so that the chain restarts after
# all of the streamers have been exhausted.
# This produces an infinite stream of data from a finite sequence
# that repeats every `N*M` steps.
# This can be used in `keras`'s `fit_generator` function
# with `validation_steps=N*M` to ensure that the validation set is
# constant at each epoch.

val_stream = pescador.ChainMux(val_streams, mode='cycle')
