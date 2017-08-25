.. _why:

Why Pescador?
=============

Pescador was developed in response to a variety of recurring problems related to data streaming for training machine learning models.
After implementing custom solutions each time these problems occurred, we converged on a set of common solutions that can be applied more broadly.
The solutions provided by Pescador may or may not fit your problem.
This section of the documentation will attempt to help you figure out if Pescador is useful for your application.


Hierarchical sampling
---------------------

`Hierarchical sampling` refers to any process where you want to sample data from a distribution by conditioning on one or more variables.
For example, say you have a distribution over real-valued observations `X` and categorical labels `Y`, and you want to sample labeled observations `(X, Y)`.
A hierarchical sampler might first select a value for `Y`, and then randomly draw an example `X` that has that label.
This is equivalent to exploiting the laws of conditional probability: :math:`P[X, Y] =
P[X|Y] \times P[Y]`.

Hierarchical sampling can be useful when dealing with highly imbalanced data, where it may sometimes be better to learn from a balanced sample and then explicitly correct for imbalance within the model.

It can also be useful when dealing with data that has natural grouping substructure beyond categories.
For example, when modeling a large collection of audio files, each file may generate multiple observations, which will all be mutually correlated.
Hierarchical sampling can be useful in neutralizing this bias during the training process.

Pescador implements hierarchical sampling via the :ref:`Mux` abstraction.
In its simplest form, `Mux` takes as input a set of :ref:`Streamer` objects from which samples are drawn randomly.
This effectively generates data by a process similar to the following pseudo-code:

.. code-block:: python
    :linenos:

    while True:
        stream_id = random_choice(streamers)
        yield next(streamers[stream_id])

The `Mux` object also lets you specify an arbitrary distribution over the set of streamers, giving you fine-grained control over the resulting distribution of samples.


The `Mux` object is also a `Streamer`, so sampling hierarchies can be nested arbitrarily deep.

Out-of-core sampling
--------------------

Another common problem occurs when the size of the dataset is too large for the machine to fit in RAM simultaneously.
Going back to the audio example above, consider a problem where there are 30,000 source files,  each of which generates 1GB of observation data, and the machine can only fit 100 source files in memory at any given time.

To facilitate this use case, the `Mux` object allows you to specify a maximum number of simultaneously active streams (i.e., the *working set*).
In this case, you would most likely implement a `generator` for each file as follows:

.. code-block:: python
    :linenos:

    def sample_file(filename):
        # Load observation data
        X = np.load(filename)

        while True:
            # Generate a random row as a dictionary
            yield dict(X=X[np.random.choice(len(X))])

    streamers = [pescador.Streamer(sample_file, fname) for fname in ALL_30K_FILES]

    for item in pescador.Mux(streamers, 100):
        model.partial_fit(item['X'])

Note that data is not loaded until the generator is instantiated.
If you specify a working set of size `k=100`, then `Mux` will select 100 streamers at random to form the working set, and only sample data from within that set.
`Mux` will then randomly evict streamers from the working set and replace them with new streamers, according to its `rate` parameter.
This results in a simple interface to draw data from all input sources but using limited memory.

`Mux` provides a great deal of flexibility over how streamers are replaced, what to do when streamers are exhausted, etc.


Parallel processing
-------------------

In the above example, all of the data I/O was handled within the `generator` function.
If the generator requires high-latency operations such as disk-access, this can become a computational bottleneck.

Pescador makes it easy to migrate data generation into a background process, so that high-latency operations do not stall the main thread.
This is facilitated by the :ref:`ZMQStreamer` object, which acts as a simple wrapper around any streamer that produces samples in the form of dictionaries of numpy arrays.
Continuing the above example:

.. code-block:: python
    :linenos:

    mux_stream = pescador.Mux(streamers, 100)

    for item in pescador.ZMQStreamer(mux_stream):
        model.partial_fit(item['X'])


Simple interface
----------------
Finally, Pescador is intended to work with a variety of machine learning frameworks, such as `scikit-learn` and `Keras`.
While many frameworks provide custom tools for handling data pipelines, each one is different, and many require using specific data structures and formats.

Pescador is meant to be framework-agnostic, and allow you to write your own data generation logic using standard Python data structures (dictionaries and numpy arrays).
We also provide helper utilities to integrate with `Keras`'s tuple generator interface.
