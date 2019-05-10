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
