#!/usr/bin/env python
"""Core classes"""
import collections
import inspect
import six
from warnings import warn

from .exceptions import PescadorError
from .util import Deprecated, rename_kw


class StreamActivator(object):
    def __init__(self, streamer):
        if not isinstance(streamer, Streamer):
            raise PescadorError("`streamer` must be / inherit from Streamer")
        self.streamer = streamer

    def __enter__(self, *args, **kwargs):
        self.streamer.activate()
        return self

    def __exit__(self, *exc):
        self.streamer.deactivate()
        return False


class Streamer(object):
    '''A wrapper class for recycling iterables and generator functions, i.e.
    streamers.

    Wrapping streamers within an object provides
    two useful features:

    1. Streamer objects can be serialized (as long as its streamer can be)
    2. Streamer objects can instantiate a generator multiple times.

    The first feature is important for parallelization (see `zmq_stream`),
    while the second feature facilitates infinite streaming from finite data
    (i.e., oversampling).


    Attributes
    ----------
    streamer : generator or iterable
        Any generator function or iterable python object.

    args : list
    kwargs : dict
        Parameters provided to `streamer`, if callable.

    Examples
    --------
    Generate random 3-dimensional vectors

    >>> def my_generator(n):
    ...     for i in range(n):
    ...         yield i
    >>> stream = Streamer(my_generator, 5)
    >>> for i in stream:
    ...     print(i)  # Displays 0, 1, 2, 3, 4


    Or with a maximum number of items

    >>> for i in stream(max_items=3):
    ...     print(i)  # Displays 0, 1, 2


    Or infinitely many examples, restarting the generator as needed

    >>> for i in stream.cycle():
    ...     print(i)  # Displays 0, 1, 2, 3, 4, 0, 1, 2, ...


    An alternate interface for the same:

    >>> for i in stream(cycle=True):
    ...     print(i)  # Displays 0, 1, 2, 3, 4, 0, 1, 2, ...

    '''

    def __init__(self, streamer, *args, **kwargs):
        '''Initializer

        Parameters
        ----------
        streamer : iterable or generator function
            Any generator function or object that is iterable when
            instantiated.

        args, kwargs
            Additional positional arguments or keyword arguments passed to
            ``streamer()`` if it is callable.

        Raises
        ------
        PescadorError
            If ``streamer`` is not a generator or an Iterable object.

        '''

        if not (inspect.isgeneratorfunction(streamer) or
                isinstance(streamer, (collections.Iterable, Streamer))):
            raise PescadorError('`streamer` must be an iterable or callable '
                                'function that returns an iterable object.')

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

        else:
            # If it's iterable, use it directly.
            self.stream_ = iter(self.streamer)

    def deactivate(self):
        self.stream_ = None

    def generate(self, max_batches=None):
        warn('`Streamer.generate(max_batches)` is deprecated in 1.1 '
             'This method will become `Streamer.iterate(max_iter)` in 2.0. '
             'Consider using this method instead, or iterating the Streamer '
             'directly (preferred), e.g. `for x in streamer:`, to maintain '
             'forwards compatibility.',
             DeprecationWarning)
        return self.iterate(max_iter=max_batches)

    def iterate(self, max_iter=None):
        '''Instantiate an iterator.

        Parameters
        ----------
        max_iter : None or int > 0
            Maximum number of iterations to yield.
            If ``None``, exhaust the stream.

        Yields
        ------
        obj : Objects yielded by the streamer provided on init.

        See Also
        --------
        cycle : force an infinite stream.

        '''
        with StreamActivator(self):
            for n, obj in enumerate(self.stream_):
                if max_iter is not None and n >= max_iter:
                    break
                yield obj

    def cycle(self):
        '''Iterate from the streamer infinitely.

        This function will force an infinite stream, restarting
        the streamer even if a StopIteration is raised.

        Yields
        ------
        obj : Objects yielded by the streamer provided on init.
        '''

        while True:
            for obj in self:
                yield obj

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
        cycle
        tuples
        keras_tuples

        '''
        warn('`Streamer.tuples()` is deprecated in 1.1 '
             'This functionality is moved to `pescador.tuples` in 2.0. '
             'Consider using this method to maintain forwards compatibility.',
             DeprecationWarning)
        if not items:
            raise PescadorError('Unable to generate tuples from '
                                'an empty item set')

        if kwargs.pop('cycle', False):
            for data in self.cycle():
                yield tuple(data[item] for item in items)
        else:
            for data in self.iterate(**kwargs):
                yield tuple(data[item] for item in items)

    def __call__(self, max_iter=None, cycle=False, max_batches=Deprecated()):
        '''Convenience interface for interacting with the Streamer.

        Parameters
        ----------
        max_iter : None or int > 0
            Maximum number of iterations to yield.
            If `None`, attempt to exhaust the stream.
            For finite streams, call iterate again, or use `cycle=True` to
            force an infinite stream.

        cycle: bool
            If `True`, cycle indefinitely.

        max_batches : None or int > 0
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `max_iter` parameter instead.
                The `max_batches` parameter will be removed in pescador 2.0.

        Yields
        ------
        obj : Objects yielded by the generator provided on init.

        See Also
        --------
        iterate
        cycle
        '''
        max_iter = rename_kw('max_batches', max_batches,
                             'max_iter', max_iter,
                             '1.1', '2.0')
        if cycle:
            gen = self.cycle()
        else:
            gen = self.iterate(max_iter=max_iter)

        for obj in gen:
            yield obj

    def __iter__(self):
        return self.iterate()
