#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

from numbers import Integral
from typing import Tuple, Callable, Optional

import numpy as np

from .typing import RandomState, Dtype


def random(
    shape: Tuple[int],
    random_state: Optional[RandomState] = None,
    data_rvs: Optional[Callable] = None,
):
    """ Generate a random jagged array.

    Args:
        shape:
            if 1D, the maximal bounds of the jagged array, otherwise the shape
        random_state:
            rng or random seed. If not given, `np.random` will be used.
        data_rvs:
            Data generation callback

    Examples:
        >>> import jagged
        >>> jagged.random((2, 2, 2), random_state=42)
        JaggedArray(data=[0.73199394 0.59865848 0.15601864],
                    shape=[[1 2]
                           [1 1]],
                    dtype=float64)
        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> jagged.random((3, 3), random_state=rng, data_rvs=lambda n: rng.randint(0, 10, n))
        JaggedArray(data=[6 3 7 ... 8 2 4],
                    shape=[[3 1 4]
                           [4 1 5]],
                    dtype=int64)
    """
    from .core import JaggedArray

    # get random state
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, Integral):
        random_state = np.random.RandomState(random_state)
    if data_rvs is None:
        data_rvs = random_state.rand

    # get shape
    shape = np.asarray(shape)
    if shape.ndim == 1:
        # max dimensions
        dim_1, *other_dims = shape
        shape = np.stack(
            [random_state.randint(1, dim + 1, dim_1) for dim in other_dims]
        )
    elif shape.ndim != 2:
        raise RuntimeError("Parameter {!r} for shape not recognised".format(shape))

    size = shape.prod(axis=0).sum()

    return JaggedArray(data_rvs(size), shape)


def infer_nan(dtype: Dtype):
    """ Infer the nan value for a given dtype

    Examples:
        >>> infer_nan(np.int32)
        nan

        >>> infer_nan(np.float64)
        nan

        >>> infer_nan(np.dtype('S4'))
        'N/A'

        >>> infer_nan(np.object_))
        None
    """

    if np.issubdtype(dtype, np.number):
        return np.nan
    elif np.issubdtype(dtype, np.str):
        return "N/A"
    else:
        return None


def is_float(obj):
    """ Whether an object is a float. """

    return isinstance(obj, (float, np.float))
