#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Deprecation utilities'''

from decorator import decorator
import inspect
import six
import warnings

from .exceptions import PescadorError


class Deprecated(object):
    '''A dummy class to catch usage of deprecated variable names'''

    def __repr__(self):
        return '<DEPRECATED parameter>'


def rename_kw(old_name, old_value, new_name, new_value,
              version_deprecated, version_removed):
    '''Handle renamed arguments.

    Parameters
    ----------
    old_name : str
    old_value
        The name and value of the old argument

    new_name : str
    new_value
        The name and value of the new argument

    version_deprecated : str
        The version at which the old name became deprecated

    version_removed : str
        The version at which the old name will be removed

    Returns
    -------
    value
        - `new_value` if `old_value` of type `Deprecated`
        - `old_value` otherwise

    Warnings
    --------
    if `old_value` is not of type `Deprecated`

    '''
    if isinstance(old_value, Deprecated):
        return new_value
    else:
        stack = inspect.stack()
        dep_func = stack[1]
        caller = stack[2]

        warnings.warn_explicit(
            "{:s}() keyword argument '{:s}' has been renamed to '{:s}' in "
            "version {:}."
            "\n\tThis alias will be removed in version "
            "{:}.".format(dep_func[3],
                          old_name, new_name,
                          version_deprecated, version_removed),
            category=DeprecationWarning,
            filename=caller[1],
            lineno=caller[2])

        return old_value


def moved(moved_from, version, version_removed):
    '''This is a decorator which can be used to mark functions
    as moved/renamed.

    It will result in a warning being emitted when the function is used.
    '''

    def __wrapper(func, *args, **kwargs):
        '''Warn the user, and then proceed.'''
        code = six.get_function_code(func)
        warnings.warn_explicit(
            "{:s}\n\tThis function was moved to '{:s}.{:s}' in "
            "version {:s}."
            "\n\tThis alias will be removed in version "
            "{:s}.".format(moved_from, func.__module__,
                           func.__name__, version, version_removed),

            category=DeprecationWarning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


def deprecated(version, version_removed):
    '''This is a decorator which can be used to mark functions
    as deprecated.

    It will result in a warning being emitted when the function is used.'''

    def __wrapper(func, *args, **kwargs):
        '''Warn the user, and then proceed.'''
        code = six.get_function_code(func)
        warnings.warn_explicit(
            "{:s}.{:s}\n\tDeprecated as of version {:s}."
            "\n\tIt will be removed in version {:s}."
            .format(func.__module__, func.__name__,
                    version, version_removed),
            category=DeprecationWarning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


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
    PescadorError
        If some two values have unequal length
    '''
    n = None

    for value in six.itervalues(batch):
        if n is None:
            n = len(value)

        elif len(value) != n:
            raise PescadorError('Unequal field lengths')

    return n
