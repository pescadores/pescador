#!/usr/bin/env python
"""Core classes"""
import collections
import inspect
import six
from .exceptions import PescadorError


class StreamActivator(object):
    def __init__(self, streamer):
        self.streamer = streamer

    def __enter__(self, *args, **kwargs):
        self.streamer.activate()
        return self

    def __exit__(self, *exc):
        self.streamer.deactivate()
        return False


class Streamer(object):
    '''A wrapper class for reusable generators.

    Wrapping generators/iterators within an object provides
    two useful features:

    1. Streamer objects can be serialized (as long as the generator can be)
    2. Streamer objects can instantiate a generator multiple times.

    The first feature is important for parallelization (see `zmq_stream`),
    while the second feature facilitates infinite streaming from finite data
    (i.e., oversampling).


    Attributes
    ----------
    generator : iterable or Streamer
        A generator function or iterable collection to draw from.
        May be another instance or subclass of Streamer.

    args : list
    kwargs : dict
        If `generator` is a function, then `args` and `kwargs`
        provide the parameters to the function.

    Examples
    --------
    Generate batches of random 3-dimensional vectors

    >>> def my_generator(n):
    ...     for i in range(n):
    ...         yield {'X': np.random.randn(1, 3)}
    >>> GS = Streamer(my_generator, 5)
    >>> for i in GS():
    ...     print(i)


    Or with a maximum number of items

    >>> for i in GS(max_items=3):
    ...     print(i)

    Or infinitely many examples, restarting the generator as needed

    >>> for i in GS.cycle():
    ...     print(i)

    An alternate interface for the same:

    >>> for i in GS(cycle=True):
    ...     print(i)
    '''

    def __init__(self, streamer, *args, **kwargs):
        '''Initializer

        Parameters
        ----------
        streamer : iterable
            Any generator function or iterable python object

        args, kwargs
            Additional positional arguments or keyword arguments to pass
            through to ``generator()``

        Raises
        ------
        PescadorError
            If ``streamer`` is not a generator or an Iterable object.
        '''

        if not (inspect.isgeneratorfunction(streamer) or
                isinstance(streamer, (collections.Iterable, Streamer))):
            raise PescadorError('streamer must be a generator, iterable, or '
                                'Streamer')

        self.streamer = streamer
        self.args = args
        self.kwargs = kwargs
        self.stream_ = None

    @property
    def active(self):
        """Returns true if the stream is active
        (ie a StopIteration) has not been thrown.
        """
        return self.stream_ is not None

    def activate(self):
        """Activates the stream."""
        if six.callable(self.streamer):
            # If it's a function, create the stream.
            self.stream_ = self.streamer(*(self.args), **(self.kwargs))

        elif isinstance(self.streamer, Streamer):
            self.stream_ = self.streamer.generate()

        else:
            # If it's iterable, use it directly.
            self.stream_ = self.streamer

    def deactivate(self):
        self.stream_ = None

    def generate(self, max_batches=None):
        '''Instantiate the generator

        Parameters
        ----------
        max_batches : None or int > 0
            Maximum number of batches to yield.
            If ``None``, exhaust the generator.
            If the stream is finite, the generator will be
            exhausted when it completes.
            Call generate again, or use cycle to force an infinite stream.

        Yields
        ------
        batch : dict
            Items from the contained generator
            If `max_batches` is an integer, then at most
            `max_batches` are generated.
        '''
        with StreamActivator(self):
            for n, x in enumerate(self.stream_):
                if max_batches is not None and n >= max_batches:
                    break
                yield x

    def cycle(self):
        '''Generates from the streamer infinitely.

        This function will force an infinite stream, restarting
        the generator even if a StopIteration is raised.

        Yields
        ------
        batch
            Items from the contained generator.
        '''

        while True:
            for item in self.generate():
                yield item

    def tuples(self, *items, **kwargs):
        '''Generate data in tuple-form instead of dicts.

        This is useful for interfacing with Keras's generator system,
        which requires iterates to be provided as tuples.

        Parameters
        ----------
        *items
            One or more dictionary keys.
            The generated tuples will correspond to
            `(batch[item1], batch[item2], ..., batch[itemk])`
            where `batch` is a single iterate produced by the
            streamer.

        cycle : bool
            If `True`, then data is generated infinitely
            using the `cycle` method.
            Otherwise, data is generated according to the
            `generate` method.

        max_batches : None or int > 0
            Maximum number of batches to yield.
            If ``None``, exhaust the generator.
            If the stream is finite, the generator will be
            exhausted when it completes.
            Call generate again, or use cycle to force an infinite stream.

        Yields
        ------
        batch : tuple
            Items from the contained generator
            If `max_batches` is an integer, then at most
            `max_batches` are generated.

        See Also
        --------
        generate
        cycle
        '''

        if not items:
            raise PescadorError('Unable to generate tuples from '
                                'an empty item set')

        if kwargs.pop('cycle', False):
            for data in self.cycle():
                yield tuple(data[item] for item in items)
        else:
            for data in self.generate(**kwargs):
                yield tuple(data[item] for item in items)

    def __call__(self, max_batches=None, cycle=False):
        '''
        Parameters
        ----------
        max_batches : None or int > 0
            Maximum number of batches to yield.
            If `None`, exhaust the generator.
            If the stream is finite, the generator will be
            exhausted when it completes.
            Call generate again, or use cycle to force an infinite stream.

        cycle: bool
            If `True`, cycle indefinitely.

        Yields
        ------
        batch : dict
            Items from the contained generator
            If `max_batches` is an integer, then at most
            `max_batches` are generated.

        See Also
        --------
        generate
        cycle
        tuples
        '''
        if cycle:
            gen = self.cycle()
        else:
            gen = self.generate(max_batches=max_batches)

        for item in gen:
            yield item
