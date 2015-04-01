#!/usr/bin/python
"""Core classes"""

import collections
import sklearn.base
import six

from . import util
from .zmq_mux import zmq_mux
from .mp_mux import threaded_mux


class Streamer(object):
    '''A wrapper class for reusable generators.

    :usage:
        >>> # make a generator
        >>> def my_generator(n):
                for i in range(n):
                    yield i
        >>> GS = Streamer(my_generator, 5)
        >>> for i in GS.generate():
                print i

        >>> # Or with a maximum number of items
        >>> for i in GS.generate(max_items=3):
                print i

    :parameters:
        - streamer : function or iterable
          Any generator function or iterable python object

        - *args, **kwargs
          Additional positional arguments or keyword arguments to pass
          through to ``generator()``

    :raises:
        - TypeError
          If ``streamer`` is not a function or an Iterable object.
    '''

    def __init__(self, streamer, *args, **kwargs):
        '''Initializer'''

        if not (hasattr(streamer, '__call__') or
                isinstance(streamer, collections.Iterable)):
            raise TypeError('streamer must be a generator or Iterable')

        self.stream = streamer
        self.args = args
        self.kwargs = kwargs

    def generate(self, max_items=None):
        '''Instantiate the generator

        :parameters:
            - max_items : None or int > 0
              Maximum number of items to yield.
              If ``None``, exhaust the generator.

        :yields:
            - Items from the contained generator
        '''

        if six.callable(self.stream):
            # If it's a function, create the stream.
            my_stream = self.stream(*(self.args), **(self.kwargs))

        else:
            # If it's iterable, use it directly.
            my_stream = self.stream

        for i, x in enumerate(my_stream):
            if max_items is None or i < max_items:
                yield x
            else:
                break


class StreamLearner(sklearn.base.BaseEstimator):
    '''A class to facilitate iterative learning from a generator.

    :parameters:
        - estimator : sklearn estimator
            The estimator to fit.  Must support the ``partial_fit`` method.

        - batch_size : int > 0
            The size of batches to be passed to ``estimator.partial_fit``.

        - max_steps : None or int > 0
            Maximum number of batch learning iterations.
            If ``None``, the learner runs until the input stream is exhausted.
    '''

    def __init__(self, estimator, batch_size=100, max_steps=None):
        ''' '''
        # Is this a supervised or unsupervised learner?
        self.supervised = isinstance(estimator, sklearn.base.ClassifierMixin)

        # Does the learner support partial fit?
        assert hasattr(estimator, 'partial_fit')

        # Is the batch size positive?
        assert batch_size > 0

        # Is the iteration bound positive or infinite?
        if max_steps is not None:
            assert max_steps > 0

        self.estimator = estimator
        self.batch_size = int(batch_size)
        self.max_steps = max_steps

    def __partial_fit(self, data, **kwargs):
        """Wrapper function to estimator.partial_fit()"""

        if self.supervised:
            args = [util.buffer_data(datum) for datum in zip(*data)]
        else:
            args = [util.buffer_data(data)]

        self.estimator.partial_fit(*args, **kwargs)

    def iter_fit(self, stream, **kwargs):
        '''Iterative learning.

        :parameters:
            - stream : iterable of (x) or (x, y)
              The data stream to fit.  Each element is assumed to be a
              single example, or a tuple of (example, label).

              Examples are collected into a batch and passed to
              ``estimator.partial_fit``.

            - kwargs
              Additional keyword arguments to ``estimator.partial_fit``.
              This is useful for things like the list of class labels for an
              SGDClassifier.

        :returns:
            - self
        '''

        # Re-initialize the model, if necessary?
        for batch in util.buffer_stream(stream, self.batch_size, self.max_steps):
            self.__partial_fit(batch, **kwargs)

        return self

    def decision_function(self, *args, **kwargs):
        '''Wrapper for estimator.predict()'''

        return self.estimator.decision_function(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        '''Wrapper for estimator.predict_proba()'''

        return self.estimator.predict_proba(*args, **kwargs)

    def predict_log_proba(self, *args, **kwargs):
        '''Wrapper for estimator.predict_log_proba()'''

        return self.estimator.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        '''Wrapper for estimator.predict()'''

        return self.estimator.predict(*args, **kwargs)

    def inverse_transform(self, *args, **kwargs):
        '''Wrapper for estimator.inverse_transform()'''

        return self.estimator.inverse_transform(*args, **kwargs)

    def transform(self, *args, **kwargs):
        '''Wrapper for estimator.transform()'''

        return self.estimator.transform(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        '''Wrapper for estimator.fit_transform()'''

        return self.estimator.fit_transform(*args, **kwargs)

    def score(self, *args, **kwargs):
        '''Wrapper for estimator.score()'''

        return self.estimator.score(*args, **kwargs)

    def fit(self, *args, **kwargs):
        '''Wrapper for estimator.fit()'''

        return self.estimator.fit(*args, **kwargs)
