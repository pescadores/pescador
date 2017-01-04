.. _example3:

Sampling from disk
==================

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
        with np.load(npz_file) as data:
            # How many rows are in the data?
            # We assume that data['Y'] has the same length
            n_total = len(data['X'])

            while True:
                # Compute the index offset
                idx = np.random.randint(n_total - n)
                yield dict(X=data['X'][idx:idx + n],
                           Y=data['Y'][idx:idx + n])

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


Memory-mapping
==============

The `NPZ` file format requires loading the entire contents of each archive into memory.
This can lead to high memory consumption when the number of active streams is large.
Note also that memory usage for each `NPZ` file will persist for as long as there is a reference to its contents.
This can be circumvented, at the cost of some latency, by copying data within the streamer function:

.. code-block:: python

    def sample_npz_copy(npz_file, n):
        with np.load(npz_file) as data:
            # How many rows are in the data?
            # We assume that data['Y'] has the same length
            n_total = len(data['X'])

            while True:
                # Compute the index offset
                idx = np.random.randint(n_total - n)
                yield dict(X=data['X'][idx:idx + n].copy(),  # <-- Note the explicit copy
                           Y=data['Y'][idx:idx + n].copy())

The above modification will ensure that memory is freed as quickly as possible.

Alternatively, *memory-mapping* can be used to only load data as needed, but requires that each array is stored in its own `NPY` file:

.. code-block:: python

    def sample_npy_mmap(npy_x, npy_y, n):

        # Open each file in "copy-on-write" mode, so that the files are read-only
        X = np.load(npy_x, mmap_mode='c')
        Y = np.load(npy_y, mmap_mode='c')

        n_total = len(X)

        while True:
            # Compute the index offset
            idx = np.random.randint(n_total - n)
            yield dict(X=X[idx:idx + n],
                       Y=Y[idx:idx + n])


    # Using this streamer is similar to the first example, but now you need a separate
    # NPY file for each X and Y
    npy_x_files = #LIST OF PRE-COMPUTED NPY FILES (X)
    npy_y_files = #LIST OF PRE-COMPUTED NPY FILES (Y)
    streams = [pescador.Streamer(sample_npz, npy_x, npy_y n)
               for (npy_x, npy_y) in zip(npy_x_files, npy_y_files)]

