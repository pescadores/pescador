#!/usr/bin/python
"""Core classes"""

import collections
import inspect
import sklearn.base
import six

from sklearn.utils.metaestimators import if_delegate_has_method


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
    generator : iterable
        A generator function or iterable collection to draw from

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
    >>> for i in GS.generate():
    ...     print(i)


    Or with a maximum number of items

    >>> for i in GS.generate(max_items=3):
    ...     print i
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
        TypeError
            If ``streamer`` is not a generator or an Iterable object.
        '''

        if not (inspect.isgeneratorfunction(streamer) or
                isinstance(streamer, collections.Iterable)):
            raise TypeError('streamer must be a generator or iterable')

        self.stream = streamer
        self.args = args
        self.kwargs = kwargs

    def generate(self, max_batches=None):
        '''Instantiate the generator

        Parameters
        ----------
        max_batches : None or int > 0
            Maximum number of batches to yield.
            If ``None``, exhaust the generator.

        Yields
        ------
        batch
            Items from the contained generator
            If `max_batches` is an integer, then at most
            `max_batches` are generated.
        '''

        if six.callable(self.stream):
            # If it's a function, create the stream.
            my_stream = self.stream(*(self.args), **(self.kwargs))

        else:
            # If it's iterable, use it directly.
            my_stream = self.stream

        for n, x in enumerate(my_stream):
            if max_batches is not None and n >= max_batches:
                break
            yield x


class StreamLearner(sklearn.base.BaseEstimator):
    '''A class to facilitate iterative learning from a generator.
    
    Attributes
    ----------
    estimator : sklearn.base.BaseEstimator
        An estimator object to wrap.  Must implement `partial_fit()`

    max_steps : None or int > 0
        The maximum number of calls to issue to `partial_fit()`.
        If `None`, run until the generator is exhausted.
    '''

    def __init__(self, estimator, max_steps=None):
        '''Learning on generators

        Parameters
        ----------
        estimator : sklearn estimator
            The estimator to fit.  Must support the ``partial_fit`` method.

        max_steps : None or int > 0
            Maximum number of batch learning iterations.
            If ``None``, the learner runs until the input stream is exhausted.
        '''
        # Does the learner support partial fit?
        assert hasattr(estimator, 'partial_fit')

        # Is this a supervised or unsupervised learner?
        self.supervised = isinstance(estimator, sklearn.base.ClassifierMixin)

        # Is the iteration bound positive or infinite?
        if max_steps is not None:
            assert max_steps > 0

        self.estimator = estimator
        self.max_steps = max_steps

    def iter_fit(self, stream, **kwargs):
        '''Iterative learning.

        Parameters
        ----------
        stream : iterable of (x) or (x, y)
            The data stream to fit.  Each element is assumed to be a
            single example, or a tuple of (example, label).

            Examples are collected into a batch and passed to
            ``estimator.partial_fit``.

        kwargs
            Additional keyword arguments to ``estimator.partial_fit``.
            This is useful for things like the list of class labels for an
            SGDClassifier.
        '''

        for i, batch in enumerate(stream):
            if self.max_steps and i >= self.max_steps:
                break

            args = [batch['X']]

            if self.supervised and 'Y' in batch:
                args.append(batch['Y'])

            elif self.supervised and 'Y' not in batch:
                raise RuntimeError('No Y-values provided by stream')

            elif 'Y' in batch:
                raise RuntimeError('Y-values supplied for '
                                   'unsupervised learner')

            self.partial_fit(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, *args, **kwargs):
        '''Wrapper for `estimator.predict()`'''

        return self.estimator.decision_function(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, *args, **kwargs):
        '''Wrapper for `estimator.predict_proba()`'''

        return self.estimator.predict_proba(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, *args, **kwargs):
        '''Wrapper for `estimator.predict_log_proba()`'''

        return self.estimator.predict(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def predict(self, *args, **kwargs):
        '''Wrapper for `estimator.predict()`'''

        return self.estimator.predict(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def inverse_transform(self, *args, **kwargs):
        '''Wrapper for `estimator.inverse_transform()`'''

        return self.estimator.inverse_transform(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def transform(self, *args, **kwargs):
        '''Wrapper for `estimator.transform()`'''

        return self.estimator.transform(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def fit_transform(self, *args, **kwargs):
        '''Wrapper for `estimator.fit_transform()`'''

        return self.estimator.fit_transform(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def score(self, *args, **kwargs):
        '''Wrapper for `estimator.score()`'''

        return self.estimator.score(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def partial_fit(self, *args, **kwargs):
        '''Wrapper for `estimator.fit()`'''
        return self.estimator.partial_fit(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator')
    def fit(self, *args, **kwargs):
        '''Wrapper for `estimator.fit()`'''

        return self.estimator.fit(*args, **kwargs)
