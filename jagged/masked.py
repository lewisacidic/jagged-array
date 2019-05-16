#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.masked
~~~~~~~~~~~~~

Support for converting jagged arrays to and from numpy masked arrays.
"""
import numpy as np

from .core import JaggedArray
from .shape import JaggedShape


def shape_to_mask(shape: JaggedShape) -> np.ndarray:
    """ the mask for a dense array for the given jagged shape.

    Args:
        shape:
            the jagged shape
    Examples:
        >>> shape_to_mask((3, (3, 2, 3)))
        array([[False, False, False],
               [False, False,  True],
               [False, False, False]])
    """
    mask = np.ones(shape.limits, dtype=bool)
    for m, shape in zip(mask, shape.to_shapes()):
        m[tuple(slice(0, dim) for dim in shape)] = False
    return mask


def dims_for_axis(mask: np.ndarray, axis=1) -> np.ndarray:
    """ get the dims of a jagged shape for a given axis of a boolean mask. """
    res = (~mask).argmin(axis=axis)
    res = res.max(axis=tuple(range(1, res.ndim))) if res.ndim >= 2 else res
    res[res == 0] = mask.shape[axis]
    return res


def mask_to_shape(mask: np.ndarray) -> JaggedShape:
    """ the jagged shape for a given dense array mask.

    Args:
        mask:
            the mask to convert to a jagged shape

    Examples:
        >>> mask_to_shape([[False, False, False], [False, False, True], [False, False, False]])
        (3, (3, 2, 3))
    """
    return JaggedShape.from_shapes(
        np.stack([dims_for_axis(mask, axis=i) for i in range(1, mask.ndim)]).T
    )


def masked_to_jagged(arr: np.ma.MaskedArray) -> JaggedArray:
    """ convert a masked array to a jagged array """
    return JaggedArray(arr.compressed(), mask_to_shape(arr.mask))


def jagged_to_masked(arr: JaggedArray) -> np.ma.MaskedArray:
    """ convert a jagged array to a masked array """
    masked = np.ma.masked_all(arr.limits, dtype=arr.dtype)
    masked[~shape_to_mask(arr.shape)] = arr.data
    return masked
