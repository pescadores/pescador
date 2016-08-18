#!/usr/bin/env python
'''Utility functions for stream manipulations

.. autosummary::
    :toctree: generated/

    batch_length

'''
import six

__all__ = ['batch_length']


def batch_length(batch):
    '''Determine the number of samples in a batch.

    Parameters
    ----------
    batch : dict
        A batch dictionary.  Each value must implement `len`.
        All values must have the same `len`.

    Returns
    -------
    n : int >= 0 or None
        The number of samples in this batch.
        If the batch has no fields, n is None.

    Raises
    ------
    RuntimeError
        If some two values have unequal length
    '''
    n = None

    for value in six.itervalues(batch):
        if n is None:
            n = len(value)

        elif len(value) != n:
            raise RuntimeError('Unequal field lengths')

    return n
