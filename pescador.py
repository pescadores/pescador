#!/usr/bin/python
"""Utilities to facilitate out-of-core learning in sklearn"""

import collections
import numpy as np
import scipy

import sklearn.base

class GeneratorSeed(object):
    '''A wrapper class for reusable generators.
    
    :usage:
        >>> # make a generator
        >>> def my_generator(n):
                for i in range(n):
                    yield i
        >>> GS = GeneratorSeed(my_generator, 5)
        >>> for i in GS.generate():
                print i
                
        >>> # Or with a maximum number of items
        >>> for i in GS.generate(max_items=3):
                print i
    
    :parameters:
        - generator : function or iterable
          Any generator function or iterable python object
          
        - *args, **kwargs
          Additional positional arguments or keyword arguments to pass through to ``generator()``
    '''
    
    def __init__(self, generator, *args, **kwargs):
        
        self.generator = generator
        self.args = args
        self.kwargs = kwargs
        
    
    def generate(self, max_items=None):
        '''Instantiate the generator
        
        :parameters:
            - max_items : None or int > 0
              Maximum number of items to yield.  If ``None``, exhaust the generator.
        '''
        
        if max_items is None:
            max_items = np.inf
        
        # If it's a function, create the stream.
        # If it's iterable, use it directly.
        
        if hasattr(self.generator, '__call__'):
            my_stream = self.generator(*(self.args), **(self.kwargs))
        elif isinstance(self.generator, collections.Iterable):
            my_stream = self.generator
        else:
            raise ValueError('generator is neither a generator nor iterable.')
            
        for i, x in enumerate(my_stream):
            if i >= max_items:
                break
            yield x

def categorical_sample(weights):
    '''Sample from a categorical distribution.
    
    :parameters:
        - weights : np.array, shape=(n,)
          The distribution to sample from.  Must be non-negative and sum to 1.0.
        
    :returns:
        - k : int in [0, n)
          The sample
    '''
    
    return np.flatnonzero(np.random.multinomial(1, weights))[0]


def generator_mux(seed_pool, n_samples, k, lam=256.0, pool_weights=None, with_replacement=True):
    
    n_seeds = len(seed_pool)
    
    # Set up the sampling distribution over streams
    seed_distribution = 1./n_seeds * np.ones(n_seeds)
    
    if pool_weights is None:
        pool_weights = seed_distribution.copy()
        
    assert len(pool_weights) == len(seed_pool)
    assert (pool_weights > 0.0).all()
    pool_weights /= np.sum(pool_weights)
    
    # If lam is not set, make the samples effectively infinite
    if lam is None:
        lam = 1e10
        
    # Instantiate the pool
    streams        = [None] * k
    
    stream_weights = np.zeros(k)
    
    for idx in range(k):
        
        if not (seed_distribution > 0).any():
            break
            
        # how many samples for this stream?
        # pick a stream
        new_idx = categorical_sample(seed_distribution)
        
        # instantiate
        streams[idx] = seed_pool[new_idx].generate(max_items=np.random.poisson(lam=lam))
        stream_weights[idx] = pool_weights[new_idx]
        
        # If we're sampling without replacement, zero out this one's probability
        if not with_replacement:
            seed_distribution[new_idx] = 0.0
            
            if (seed_distribution > 0).any():
                seed_distribution[:] /= np.sum(seed_distribution)
        
    Z = np.sum(stream_weights)
    
    
    # Main sampling loop
    n = 0
    
    while n < n_samples and Z > 0.0:
        # Pick a stream
        idx = categorical_sample(stream_weights / Z)
        
        # Can we sample from it?
        try:
            # Then yield the sample
            yield streams[idx].next()
            
            # Increment the sample counter
            n = n + 1
            
        except StopIteration:
            # Oops, this one's exhausted.  Replace it and move on.
            
            # Are there still kids in the pool?  Okay.
            if (seed_distribution > 0).any():
            
                new_idx = categorical_sample(pool_weights)
            
                streams[idx] = seed_pool[new_idx].generate(max_items=np.random.poisson(lam=lam))
                stream_weights[idx] = pool_weights[new_idx]
                
                # If we're sampling without replacement, zero out this one's probability and renormalize
                if not with_replacement:
                    seed_distribution[new_idx] = 0.0
                    
                    if (seed_distribution > 0).any():
                        seed_distribution[:] /= np.sum(seed_distribution)
                
            else:
                # Otherwise, this one's exhausted.  Set its probability to 0 and keep going
                stream_weights[idx] = 0.0
                
            Z = np.sum(stream_weights)

def stream_fit(estimator, data_sequence, batch_size=100, max_steps=None, **kwargs):
    '''Fit a model to a generator stream.
    
    :parameters:
      - estimator : sklearn.base.BaseEstimator
        The model object.  Must implement ``partial_fit()``
      
      - data_sequence : generator
        A generator that yields samples
    
      - batch_size : int
        Maximum number of samples to buffer before updating the model
      
      - max_steps : int or None
        If ``None``, run until the stream is exhausted.
        Otherwise, run until at most ``max_steps`` examples have been processed.
    '''
    
    # Is this a supervised or unsupervised learner?
    supervised = isinstance(estimator, sklearn.base.ClassifierMixin)
    
    # Does the learner support partial fit?
    assert(hasattr(estimator, 'partial_fit'))
    
    def _matrixify(data):
        """Determine whether the data is sparse or not, act accordingly"""

        if scipy.sparse.issparse(data[0]):
            n = len(data)
            d = np.prod(data[0].shape)
    
            data_s = scipy.sparse.lil_matrix((n, d), dtype=data[0].dtype)
    
            for i in range(len(data)):
                idx = data[i].indices
                data_s[i, idx] = data[i][:, idx]

            return data_s.tocsr()
        else:
            return np.asarray(data)

    def _run(data, supervised):
        """Wrapper function to partial_fit()"""

        if supervised:
            args = map(_matrixify, zip(*data))
        else:
            args = [_matrixify(data)]

        estimator.partial_fit(*args, **kwargs)
            
    buf = []
    for i, x_new in enumerate(data_sequence):
        buf.append(x_new)
        
        # We've run too far, stop
        if max_steps is not None and i > max_steps:
            break
        
        # Buffer is full, do an update
        if len(buf) == batch_size:
            _run(buf, supervised)
            buf = []
    
    # Update on whatever's left over
    if len(buf) > 0:
        _run(buf, supervised)
