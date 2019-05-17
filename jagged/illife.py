#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.illife
~~~~~~~~~~~~~

Support for converting jagged arrays to and from Illife vectors.
"""
import numpy as np

from .core import JaggedArray
from .typing import IllifeLike


def sanitize_illife(obj: IllifeLike):
    """ Sanitize an illife-like array

    Args:
        ivec:
            object to convert into an illife vector.

    Examples:
        >>> sanitize_illife([[0, 1], [2]])
        array([array([0, 1]), array([2])])

        >>> sanitize_illife(((0, 1), (2)))
        array([array([0, 1]), array([2])])

        >>> sanitize_illife([[[0, 1], [2, 3]], [[4, 5, 6], [7, 8, 9]])
        array([array([[0, 1], [2, 3]]), array([[4, 5, 6], [7, 8, 9]])])
     """
    # avoid attempting to broadcast for subarrays with first dim equal to 1
    ivec = np.empty(len(obj), dtype=object)
    ivec[...] = [np.array(arr) for arr in obj]
    return ivec


def illife_to_jagged(ivec: IllifeLike):
    """ Convert an Illife vector to a jagged array. """

    ivec = sanitize_illife(ivec)
    return JaggedArray(
        np.concatenate([arr.flatten() for arr in ivec]),
        shapes=np.array([arr.shape for arr in ivec]),
    )


def jagged_to_illife(jarr: JaggedArray, copy: bool = False):
    """ Convert a jagged array to an illife vector. """

    if copy:
        jarr = jarr.copy()
    return sanitize_illife(jarr)
