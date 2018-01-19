#! -*- coding: utf-8 -*-
"""
Hierarchical sampling with ShuffledMux
======================================

Hierarchical sampling can be done in pescador by composing mux objects.
This example will illustrate how to use different types of muxen to
achieve different effects.

As a motivating example, consider a machine learning problem where
training data are images belonging to 100 classes, such as `car`, `boat`,
`pug`, etc.
For each class, you have a list of images belonging to that class.
Perhaps the total number of images is larger than fits comfortably
in memory, but you still want to produce a stream of training data
with uniform class presentation.

To solve this in pescador, we will first create a `PoissonMux` for each
sub-population (e.g., one for cars, one for boats, etc.), which will
maintain a small active set.
We will then combine those sub-population muxen using `ShuffledMux` to
produce the output stream.

"""

# Code source: Brian McFee
# License: BSD 3 Clause

from __future__ import print_function
import pescador

#####################
# Setup
#####################
# We'll demonstrate this with a simpler problem
# involving two populations of streamers:
#
# - Population 1 generates upper-case letters
# - Population 2 generates lower-case letters

# First, let's make a simple generator that makes an infinite
# sequence of a given letter.
def letter(c):
    while True:
        yield c

# Let's make the two populations of streamers
pop1 = [pescador.Streamer(letter, c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
pop2 = [pescador.Streamer(letter, c) for c in 'abcdefghijklmnopqrstuvwxyz']

# We'll sample population 1 with 3 streamers active at any time.
# Each streamer will generate, on average, 5 samples before being
# replaced.
mux1 = pescador.PoissonMux(pop1, 3, 5)

# Let's have 5 active streamers for population 2, and replace
# them after 2 examples on average.
mux2 = pescador.PoissonMux(pop2, 5, 2)

####################
# Mux composition
####################
# We multiplex the two populations using a ShuffledMux.
# The ShuffledMux keeps all of its input streamers active,
# and draws samples independently at random from each one.

# This should generate an approximately equal number of upper- and
# lower-case letters, with more diversity among the lower-case letters.
hier_mux = pescador.ShuffledMux([mux1, mux2])
print(''.join(hier_mux(max_iter=80)))

#####################
# Weighted sampling
#####################
# If you want to specify the sampling probability of mux1 and mux2,
# you can supply weights to the ShuffledMux.
# By default, each input is equally likely.

# This should generate three times as many upper-case as lower-case letters.
weight_mux = pescador.ShuffledMux([mux1, mux2], weights=[0.75, 0.25])
print(''.join(weight_mux(max_iter=80)))
