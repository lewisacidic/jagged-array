#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.utils
~~~~~~~~~~~~

Utility functions for the jagged-array project.
"""
from typing import Any

import numpy as np

from .typing import DtypeLike
from .typing import JaggedShapeLike


def infer_nan(dtype: DtypeLike) -> Any:
    """ Infer the nan value for a given dtype

    Examples:
        >>> infer_nan(np.int32)
        nan

        >>> infer_nan(np.float64)
        nan

        >>> infer_nan(np.dtype('U4'))
        'nan'

        >>> infer_nan(np.dtype('S4'))
        b'nan'

        >>> infer_nan(np.object_))
        nan
    """

    return np.array(np.nan).astype(dtype).item()


def is_float(obj: Any) -> bool:
    """ Whether an object is a float. """

    return isinstance(obj, (float, np.float))


def shape_to_shapes(shape):
    """ Convert a jagged shape to shapes

    Examples:
        >>> shape_to_shapes((3, (1, 3, 2)))
        array([[1],
               [3],
               [2]])

        >>> shape_to_shapes((3, (3, 1, 1), 2))
        array([[3, 2],
               [1, 2],
               [1, 2]])

        >>> shape_to_shapes((4, (1, 3, 2)))
        Traceback (most recent call last):
            ...
        ValueError: invalid shape. Jagged axes must have number of entries equal to length of inducing dim
    """

    inducing, *axs = shape
    if any(len(ax) != inducing for ax in axs if isinstance(ax, tuple)):
        msg = "invalid shape. Jagged axes must have number of entries equal to length of inducing dim"
        raise ValueError(msg)
    res = np.empty((inducing, len(axs)), dtype=int)
    for i, ax in enumerate(axs):
        res[:, i] = ax
    return res


def shapes_to_shape(shapes: np.ndarray) -> JaggedShapeLike:
    """ Convert an array of shapes to a jagged shape.

    Args:
        shapes:
            An array of shapes

    Examples:
        >>> shapes_to_shape([[1], [3], [2]])
        (3, (1, 3, 2))

        >>> shapes_to_shape([[1, 3], [3, 2], [2, 3]])
        (3, (1, 3, 2), (3, 2, 3))

        >>> shapes_to_shape([[3, 2], [1, 2], [1, 2]])
        (3, (3, 1, 1), 2)

    See Also:
        shape_to_shapes: the reverse of this function
    """
    shapes = np.asarray(shapes)
    rest = (dim[0] if (dim == dim[0]).all() else tuple(dim) for dim in shapes.T)
    return (len(shapes), *rest)


def shapes_to_size(shapes: JaggedShapeLike) -> int:
    """ Get the number of entries for an array of shapes.

    Args:
        shapes:
            An array of shapes

    Examples:
        >>> size_from_shape((2, (1, 2)))
        3

        >>> size_from_shape((3, (2, 3, 2), 2))
        14

        >>> size_from_shape((3, (2, 3, 2), (1, 4, 2)))
        18

        >>> size_from_shape((3, 2, (2, 3, 2), (1, 4, 2)))
        36
    """
    return shapes.prod(axis=1).sum()
