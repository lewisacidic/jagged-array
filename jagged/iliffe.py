#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.iliffe
~~~~~~~~~~~~~

Support for converting jagged arrays to and from Iliffe vectors.
"""
import numpy as np

from .core import JaggedArray
from .typing import DtypeLike
from .typing import IliffeLike


def sanitize_iliffe(obj: IliffeLike):
    """ Sanitize an iliffe-like array

    Args:
        ivec:
            object to convert into an iliffe vector.

    Examples:
        >>> sanitize_iliffe([[0, 1], [2]])
        array([array([0, 1]), array([2])])

        >>> sanitize_iliffe(((0, 1), (2)))
        array([array([0, 1]), array([2])])

        >>> sanitize_iliffe([[[0, 1], [2, 3]], [[4, 5, 6], [7, 8, 9]])
        array([array([[0, 1], [2, 3]]), array([[4, 5, 6], [7, 8, 9]])])
     """
    # avoid attempting to broadcast for subarrays with first dim equal to 1
    ivec = np.empty(len(obj), dtype=object)
    ivec[...] = [np.array(arr) for arr in obj]
    return ivec


def iliffe_to_jagged(ivec: IliffeLike, dtype: DtypeLike = None):
    """ Convert an Illife vector to a jagged array. """

    ivec = sanitize_iliffe(ivec)
    return JaggedArray(
        np.concatenate([arr.flatten() for arr in ivec]),
        shapes=np.array([arr.shape for arr in ivec]),
        strides=np.array([arr.strides for arr in ivec]),
        dtype=dtype,
    )


def jagged_to_iliffe(jarr: JaggedArray, copy: bool = False):
    """ Convert a jagged array to an iliffe vector. """

    if copy:
        jarr = jarr.copy()
    return sanitize_iliffe(jarr)
