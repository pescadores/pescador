#!/usr/bin/env python
'''Buffered streamers'''
from warnings import warn

from . import core
from .maps import buffer_stream
from . import util


class BufferedStreamer(core.Streamer):
    """Buffers a stream into batches of examples.

    Examples
    --------
    >>> def my_generator(n):
    ...     # Generates a single 30-dimensional example vector for each iterate
    ...     for i in range(n):
    ...         yield dict(X=np.random.randn(1, 30))
    >>> # Wrap the generator in a Streamer
    >>> S = pescador.Streamer(my_generator, 128)
    >>> # A buffered streamer will combine N iterates into a single batch
    >>> N = 10
    >>> B = pescador.BufferedStreamer(my_generator, N)
    >>> for batch in B:
    ...     # Work on a batch of N=10 examples
    ...     MY_PROCESS_FUNCTION(batch)
    """

    def __init__(self, streamer, buffer_size,
                 strict_batch_size=True):
        """
        Parameters
        ----------
        streamer : pescador.Streamer
            A `Streamer` object to sample from

        buffer_size : int
            Number of samples to buffer into a batch.

        strict_batch_size : bool
            If `True`, will only return batches of length `buffer_size`.
             If the enclosed streamer runs out of samples before completing
             the last batch, generate() will raise a StopIteration
             instead of returning a partial batch.

            If `False`, if the enclosed streamer runs out of samples before
             completing teh last batch, will just return the number
             of samples currently in the buffer.
        """
        warn('`BufferedStreamer` is deprecated in 1.1 '
             'This functionality is superseded by the generator function '
             '`pescador.buffer_data` in 2.0, which can be used with '
             '`pescador.Streamer` to similar ends.'
             'Use this idiom instead to maintain forwards compatibility.',
             DeprecationWarning)
        self.streamer = streamer
        if not isinstance(streamer, core.Streamer):
            self.streamer = core.Streamer(streamer)
        self.buffer_size = buffer_size
        self.strict_batch_size = strict_batch_size

    def activate(self):
        """Activates the stream."""
        self.stream_ = self.streamer

    def iterate(self, max_iter=None, partial=True):
        """Generate samples from the streamer.

        Parameters
        ----------
        max_iter : int
            For the BufferedStreamer, max_iter is the
            number of *buffered* batches that are generated,
            not the number of individual samples.

        partial : bool, default=True
            If True, will return a final batch smaller than the requested size.
        """
        with core.StreamActivator(self):
            for n, batch in enumerate(buffer_stream(self.stream_,
                                                    self.buffer_size,
                                                    partial=partial)):
                if max_iter is not None and n >= max_iter:
                    break
                yield batch


batch_length = util.moved('pescador.util.batch_length',
                          '1.1', '2.0')(util.batch_length)

buffer_batch = util.moved('pescador.maps.buffer_stream',
                          '1.1', '2.0')(buffer_stream)
