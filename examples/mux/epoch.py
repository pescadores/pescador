#! -*- coding: utf-8 -*-
"""
Using cycle mode to create data epochs
======================================

In some machine learning applications, it is common to train
in *epochs* where each epoch consists of a full pass through
the data set.
There are a few ways to produce this behavior with pescador,
depending on the exact sampling properties you have in mind.

- If presentation order does not matter, and a deterministic
  sequence is acceptable, then this can be achieved with
  `ChainMux` as demonstrated in :ref:`Using ChainMux for repeatable streams`.
  This is typically a good approach for validation or evaluation,
  but not training, since the deterministic sequence order could bias
  the model.

- If you want random presentation order, but want to ensure that
  all data is touched once per epoch, then the `StochasticMux` can be
  used in `cycle` mode to restart all streamers once they've been
  exhausted.
"""

# Imports
import numpy as np
import pescador

########################
# Setup
########################
# We'll assume that the data lives in some N files
# For convenience, we'll assume that each file produces M examples.
# Our goal is to ensure that each example is generated once per epoch.

files = ['file1.npz', 'file2.npz', 'file3.npz']
N = len(files)

M = 10  # or whatever the number of examples per file is

########################
# Data generator
########################
# We'll make a simple generator that streams the first m
# examples from a given file.
# The npz file is assumed to store two arrays: X and Y
# containing inputs and outputs.
# Once the streamer produces m examples, it exits.
#
# Here, we'll use the decorator interface to declare this
# generator as a pescador Streamer


@pescador.streamable
def data_gen(filename, m):
    data = np.load(filename)
    X = data['X']
    Y = data['Y']
    for i in range(m):
        yield dict(X=X[i], y=Y[i])

###############################
# Constructing the streamers
###############################
# We'll make a streamer for each source file

streams = [data_gen(fn, M) for fn in files]

###############################
# Epochs with StochasticMux
###############################
# The `StochasticMux` has three modes of operation, which control
# how its input streams are activated and replaced:
# - `mode='with_replacement'` allows each streamer to be activated
#   multiple times, even simultaneously.
# - `mode='single_active'` does not allow a streamer to be active
#   more than once at a time, but an inactive streamer can be activated
#   at any time.
# - `mode='exhaustive'` is like `single_active`, but does not allow
#   previously used streamers to be re-activated.
#
# For epoch-based sampling, we will use `exhaustive` mode to ensure
# that streamers are not reactivated within the epoch.
#
# Since each data stream produces exactly `M` examples, this would lead
# to a finite sample stream (i.e., only one epoch).
# To prevent the mux from exiting after the first epoch, we'll use `cycle` mode.
#

k = 100  # or however many streamers you want simultaneously active

# We'll use `rate=None` here so that the number of samples per stream is
# determined by the streamer (`M`) and not the mux.

mux = pescador.StochasticMux(streams, k, rate=None, mode='exhaustive')

epoch_stream = mux(cycle=True)

####################
# The `epoch_stream` will produce an infinite sequence of iterates.
# The same samples are presented (in random order) in the
# first `N*M`, second `N*M`, etc. disjoint sub-sequences, each
# of which may be considered as an *epoch*.
#
# *NOTE*: for this approach to work with something like `keras`'s
# `fit_generator` method, you need to be able to explicitly calculate
# the duration of an epoch, which  means that the number of samples
# per streamer (`M` here) must be known in advance.
#
