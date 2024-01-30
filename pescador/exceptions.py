#!/usr/bin/env python
"""Exception classes for pescador"""


class PescadorError(Exception):
    """The root pescador exception class"""

    pass


class DataError(PescadorError):
    """Exception when expecting "data" objects, e.g. {key: npndarray}"""

    pass
