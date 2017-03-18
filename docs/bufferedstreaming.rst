.. _bufferedstreaming:

Buffered Streaming
==================

In a machine learning setting, it is common to train a model with multiple input datapoints simultaneously, in what are commonly referred to as "minibatches". To achieve this, pescador provides the :ref:`BufferedStreamer`, which will "buffer" your batches into fixed batch sizes.

Following up on the first example, we use the `noisy_samples` generator.

.. code-block:: python
    :linenos:

    import pescador

    # Create an initial streamer
    streamer = pescador.Streamer(noisy_samples, X[train], Y[train])

    minibatch_size = 128
    # Wrap your streamer
    buffered_streamer = pescador.BufferedStreamer(streamer, minibatch_size)

    # Generate batches in exactly the same way as you would from the base streamer
    for batch in buffered_streamer():
        ...

A few important points to note about using :ref:`BufferedStreamer`:

    - :ref:`BufferedStreamer` will concatenate your arrays, such that the first dimension contains the number of batches (`minibatch_size` in the above example.

    - Each key in the batches generated will be concatenated (across all the batches buffered).

    - A consequence of this is that you must make sure that your generators yield batches such that every key contains arrays shaped (N, ...), where N is the number of batches generated.
