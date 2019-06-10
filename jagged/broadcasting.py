#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.broadcasting
~~~~~~~~~~~~~~~~~~~

Broadcasting support for jagged-array.
"""
from itertools import zip_longest

from .core import JaggedArray


def broadcast_to(arr, shape):
    """ Broadcast a jagged array to a given shape. """

    # get right ndims
    if isinstance(arr, JaggedArray):
        if arr.shape == shape:
            return arr
        else:
            raise NotImplementedError


def dims_compatible(a, b):
    """ whether dimensions are broadcast compatible

    Examples:
        >>> dims_compatible(1, 3)
        True

        >>> dims_compatible((3, 2, 3), 1)
        True

        >>> dims_compatible(3, 2)
        False

        >>> dims_compatible((3, 2, 3), (3, 2, 3))
        True

        >>> dims_compatible((3, 2, 3), (2, 3, 2))
        False
    """

    return (a == b) or (a == 1) or (b == 1)


def broadcast_dim(a, b):
    """ Broadcast compatible dimensions

    Examples:
        >>> broadcast_dim(3, 1)
        3

        >>> broadcast_dim(1, (3, 2, 3)))
        (3, 2, 3)

        >>> broadcast_dim(3, 3)
        3
    """
    if a == b:
        return a
    elif a == 1:
        return b
    elif b == 1:
        return a


def broadcast_shapes(*shapes):
    """ broadcast shapes together

    Examples:
        >>> broadcast_shapes((1,), (3,))
        (3,)

        >>> broadcast_shapes((1, 3), (2, 1))
        (2, 3)

        >>> broadcast_shapes((3, 2), (1, 2), (1, 1))
        (3, 2)

        >>> broadcast_shapes((3, 1), (3, (3, 2, 3)))
        (3, (3, 2, 3))

        >>> broadcast_shape((3, 1, (3, 2, 3)), (3, 2, 1))
        (3, 2, (3, 2, 3))

        # broadcast jagged to dense
        >>> broadcast_shape((3, 2), (3, 1, (2, 1, 2)))
        (3, 2)
    """

    if len(shapes) == 1:
        return shapes[0]
    elif len(shapes) == 2:
        s1, s2 = shapes
        sr1, sr2 = s1[::-1], s2[::-1]
        if not all(dims_compatible(a, b) for a, b in zip(sr1, sr2)):
            msg = f"operands cound not be broadcast together with shapes {s1} {s2}"
            raise ValueError(msg)
        return tuple(
            broadcast_dim(m, n) for m, n in zip_longest(sr1, sr2, fillvalue=1)
        )[::-1]
    else:
        return broadcast_shapes([shapes[0], broadcast_shapes(shapes[1:])])


def broadcast_arrays(*args):
    shape = broadcast_shapes(*(args.shape for args in args))
    return tuple(broadcast_to(arr, shape) for arr in args)
