#!/usr/bin/env python
"""Core classes"""
import collections
import inspect
import six
from .exceptions import PescadorError


class StreamActivator(object):
    def __init__(self, streamer):
        for mname in ['activate', 'deactivate']:
            if not hasattr(streamer, mname):
                raise PescadorError(
                    "`streamer` doesn't implement the Streamer interface: no "
                    "attribute `{}`".format(mname))
        self.streamer = streamer

    def __enter__(self, *args, **kwargs):
        self.streamer.activate()
        return self

    def __exit__(self, *exc):
        self.streamer.deactivate()
        return False


class Streamer(object):
    '''A wrapper class for reusable iterators.

    Wrapping iterators/generators within an object provides
    two useful features:

    1. Streamer objects can be serialized (as long as the iterator can be)
    2. Streamer objects can instantiate a generator multiple times.

    The first feature is important for parallelization (see `zmq_stream`),
    while the second feature facilitates infinite streaming from finite data
    (i.e., oversampling).


    Attributes
    ----------
    iterator : iterator or generator
        A callable iterator, function, or generator that yields stuff.

    args : list
    kwargs : dict
        If `iterator` is a function, then `args` and `kwargs`
        provide the parameters to the function.

    Examples
    --------
    Generate random 3-dimensional vectors

    >>> def my_generator(n):
    ...     for i in range(n):
    ...         yield np.random.randn(1, 3)
    >>> stream = Streamer(my_generator, 5)
    >>> for i in stream:
    ...     print(i)


    Or with a maximum number of items

    >>> for i in stream(max_items=3):
    ...     print(i)

    Or infinitely many examples, restarting the generator as needed

    >>> for i in stream.cycle():
    ...     print(i)

    An alternate interface for the same:

    >>> for i in stream(max_items=10, cycle=True):
    ...     print(i)
    '''

    def __init__(self, iterator, *args, **kwargs):
        '''Initializer

        Parameters
        ----------
        iterator : callable
            Any generator function or object that is iterable when
            instantiated.

        args, kwargs
            Additional positional arguments or keyword arguments to pass
            through to ``iterator()``

        Raises
        ------
        PescadorError
            If ``iterator`` is not a generator or an Iterable object.
        '''

        if not (inspect.isgeneratorfunction(iterator) or
                isinstance(iterator, (collections.Iterable, Streamer))):
            raise PescadorError('`iterator` must be a generator, iterator, or '
                                'Streamer')

        # TODO: Button this up based on discussion of #75
        self.streamer = iterator
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
        # TODO: Button this up based on discussion of #75
        if six.callable(self.streamer):
            # If it's a function, create the stream.
            self.stream_ = self.streamer(*(self.args), **(self.kwargs))

        elif isinstance(self.streamer, Streamer):
            self.stream_ = self.streamer.generate()

        else:
            # If it's iterable, use it directly.
            # TODO: Remove this.
            self.stream_ = self.streamer

    def deactivate(self):
        self.stream_ = None

    def generate(self, max_iter=None):
        '''Instantiate the generator

        Parameters
        ----------
        max_iter : None or int > 0
            Maximum number of iterations to yield.
            If ``None``, exhaust the generator.
            If the stream is finite, the generator will be
            exhausted when it completes.

        Yields
        ------
        obj : Objects yielded by the iterator provided on init.

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
        '''Generates from the streamer infinitely.

        This function will force an infinite stream, restarting
        the iterator even if a StopIteration is raised.

        Yields
        ------
        obj : Objects yielded by the iterator provided on init.
        '''

        while True:
            for obj in self.generate():
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
        generate
        cycle
        '''
        # TODO: Remove from class, turn into transform
        if not items:
            raise PescadorError('Unable to generate tuples from '
                                'an empty item set')

        if kwargs.pop('cycle', False):
            for data in self.cycle():
                yield tuple(data[item] for item in items)
        else:
            for data in self.generate(**kwargs):
                yield tuple(data[item] for item in items)

    def __call__(self, max_iter=None, cycle=False):
        '''Convenience interface for interacting with the Streamer.

        Parameters
        ----------
        max_iter : None or int > 0
            Maximum number of iterations to yield.
            If `None`, exhaust the iterator.
            If the stream is finite, the iterator will be
            exhausted when it completes.
            Call generate again, or use cycle to force an infinite stream.

        cycle: bool
            If `True`, cycle indefinitely.

        Yields
        ------
        obj : Objects yielded by the iterator provided on init.

        See Also
        --------
        generate
        cycle
        tuples
        '''
        if cycle:
            gen = self.cycle()
        else:
            gen = self.generate(max_iter=max_iter)

        for obj in gen:
            yield obj

    def __iter__(self):
        return self.generate()
