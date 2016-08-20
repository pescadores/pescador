import numpy as np

from . import core
from . import util


class BufferedStreamer(core.Streamer):
    """Buffers a stream into batches.

    """
    def __init__(self, streamer, buffer_size,
                 strict_batch_size=True):
        """
        Parameters
        ----------
        streamer : pescador.Streamer
            Expects a Streamer, but if you pass it an iterable,
            it will just wrap it in a streamer.

        buffer_size : int
            Number of samples to buffer into a batch.

        strict_batch_size : bool
            If True, will only return batches of length buffer_size.
             If the enclosed streamer runs out of samples before completing
             the last batch, generate() will raise a StopIteration
             instead of returning a partial batch.
            If False, if the enclosed streamer runs out of samples before
             completing teh last batch, will just return the number
             of samples currently in the buffer.
        """
        self.streamer = streamer
        if not isinstance(streamer, core.Streamer):
            self.streamer = core.Streamer(streamer)
        self.buffer_size = buffer_size
        self.strict_batch_size = strict_batch_size

    def activate(self):
        """Activates the stream."""
        self.stream_ = self.streamer

    def generate(self, max_batches=None):
        """
        Parameters
        ----------
        max_batches : int
            For the BufferedStreamer, max_batches is the
            number of *buffered* batches that are generated,
            not the number of individual samples.
        """
        with core.StreamActivator(self):
            for n, batch in enumerate(buffer_batch(self.stream_.generate(),
                                                   self.buffer_size)):
                if max_batches is not None and n >= max_batches:
                    break
                yield batch


def buffer_batch(generator, buffer_size):
    '''Buffer an iterable of batches into larger (or smaller) batches

    Parameters
    ----------
    generator : iterable
        The generator to buffer

    buffer_size : int > 0
        The number of examples to retain per batch.

    Yields
    ------
    batch
        A batch of size at most `buffer_size`
    '''

    batches = []
    n = 0

    for x in generator:
        batches.append(x)
        n += util.batch_length(x)

        if n < buffer_size:
            continue

        batch, batches = __split_batches(batches, buffer_size)

        if batch is not None:
            yield batch
            batch = None
            n = 0

    # Run out the remaining samples
    while batches:
        batch, batches = __split_batches(batches, buffer_size)
        if batch is not None:
            yield batch


def __split_batches(batches, buffer_size):
    '''Split at most one batch off of a collection of batches.

    Parameters
    ----------
    batches : list
        List of batch objects

    buffer_size : int > 0 or None
        Size of the desired buffer.
        If None, the entire stream is exhausted.

    Returns
    -------
    batch, remaining_batches
        One batch of size up to buffer_size,
        and all remaining batches.

    '''

    batch_size = 0
    batch_data = []

    # First, pull off all the candidate batches
    while batches and (buffer_size is None or
                       batch_size < buffer_size):
        batch_data.append(batches.pop(0))
        batch_size += util.batch_length(batch_data[-1])

    # Merge the batches
    batch = dict()
    residual = dict()

    has_residual = False
    has_data = False

    for key in batch_data[0].keys():
        batch[key] = np.concatenate([data[key] for data in batch_data])

        residual[key] = batch[key][buffer_size:]

        if len(residual[key]):
            has_residual = True

        # Clip to the appropriate size
        batch[key] = batch[key][:buffer_size]

        if len(batch[key]):
            has_data = True

    if has_residual:
        batches.insert(0, residual)

    if not has_data:
        batch = None

    return batch, batches