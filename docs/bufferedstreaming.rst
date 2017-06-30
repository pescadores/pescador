.. _bufferedstreaming:

Buffered Streaming
==================

In a machine learning setting, it is common to train a model with multiple input datapoints simultaneously, in what are commonly referred to as "minibatches". To achieve this, pescador provides the :ref:`pescador.maps.buffer_stream` map transformer, which will "buffer" a data stream into fixed batch sizes.

Following up on the first example, we use the `noisy_samples` generator.

.. code-block:: python
    :linenos:

    import pescador

    # Create an initial streamer
    streamer = pescador.Streamer(noisy_samples, X[train], Y[train])

    minibatch_size = 128
    # Wrap your streamer
    buffered_sample_gen = pescador.buffer_stream(streamer, minibatch_size)

    # Generate batches in exactly the same way as you would from the base streamer
    for batch in buffered_sample_gen:
        ...



A few important points to note about using :ref:`pescador.maps.buffer_stream`:

    - :ref:`pescador.maps.buffer_stream` will concatenate your arrays, adding a new sample dimension such that the first dimension contains the number of batches (`minibatch_size` in the above example). e.g. if your samples are shaped (4, 5), a batch size of 10 will produce arrays shaped (10, 4, 5)

    - Each key in the batches generated will be concatenated (across all the samples buffered).

    - `pescador.maps.buffer_stream`, like all `pescador.maps` transformers, returns a *generator*, not a Streamer. So, if you still want it to behave like a streamer, you have to wrap it in a streamer. Following up on the previous example:

.. code-block:: python
    :linenos:
    
    batch_streamer = pescador.Streamer(buffered_sample_gen)

    # Generate batches as a streamer:
    for batch in batch_streamer:
        # batch['X'].shape == (minibatch_size, ...)
        # batch['Y'].shape == (minibatch_size,)
        ...


    # Or, another way:
    batch_streamer = pescador.Streamer(pescador.buffer_stream, streamer, minibatch_size)
