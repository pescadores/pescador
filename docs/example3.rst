.. _example3:

Sampling from NPZ archives
==========================

A common use case for `pescador` is to sample data from a large collection of existing archives.
As a concrete example, consider the problem of fitting a statistical model to a large
corpus of musical recordings.
When the corpus is sufficiently large, it is impossible to fit the entire set in memory
while estimating the model parameters.
Instead, one can pre-process each song to store pre-computed features (and, optionally,
target labels) in a *numpy zip* `NPZ` archive.
The problem then becomes sampling data from a collection of `NPZ` archives.

Here, we will assume that the pre-processing has already been done so that each `NPZ` file contains a numpy array of features `X` and labels `Y`.
We will define infinite samplers that pull `n` examples per iterate.

.. code-block:: python

    import numpy as np
    import pescador

    def sample_npz(npz_file, n):
        '''Generate an infinite sequence of contiguous samples
        from the input `npz_file`.

        Each iterate has `n > 0` rows.
        '''
        # We use the context form of data loading here
        # so that memory is released when the streamer is deactivated
        with np.load(npz_file) as data:
            # How many rows are in the data?
            # We assume that data['Y'] has the same length
            n_total = len(data['X'])

            while True:
                # Compute the index offset
                idx = np.random.randint(n_total - n)
                yield dict(X=data['X'][idx:idx+n],
                           Y=data['Y'][idx:idx+n])

Applying the `sample_npz` function above to a list of `npz_files`, we can make a
multiplexed streamer object as follows:

.. code-block:: python

    n = 16
    npz_files = #LIST OF PRE-COMPUTED NPZ FILES
    streams = [pescador.Streamer(sample_npz, npz_f, n) for npz_f in npz_files]

    # Keep 32 streams alive at once
    # Draw on average 16 patches from each stream before deactivating
    mux_stream = pescador.Mux(streams, k=32, lam=16)

    for batch in mux_stream.generate(max_batches=1000):
        # DO LEARNING HERE
        pass
