#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.api
~~~~~~~~~~

Top level function for operating on jagged arrays.
"""
from typing import Callable
from typing import Iterable
from typing import Optional

import numpy as np

from .core import JaggedArray
from .typing import ArrayLike
from .typing import AxisLike
from .typing import DtypeLike
from .typing import JaggedShapeLike
from .typing import RandomState


def zeros(shape: JaggedShapeLike, dtype: Optional[DtypeLike] = None):
    """ Create a jagged array of zeros in a given jagged shape.

    Args:
        shape:
            the shape of the array to produce

        dtype:
            the dtype of the array to produce

    Examples:
        >>> zeros((3, (3, 2, 3)))
        JaggedArray([0, 0, 0,...], shape=(3, (3, 2, 3)))

        >>> zeros((2, 2, (1, 2)), dtype=bool)
        JaggedArray([False, False, ...], shape=(2, 2, (1, 2)))

     """
    raise NotImplementedError


def zeros_like(arr: JaggedArray, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create a jagged array of zeros in the shape of another jagged array.

    Args:
        arr:
            the jagged array to use as template.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> zeros_like(JaggedArray(np.arange(8), (3, (3, 2, 3))))
        JaggedArray([0, 0, ...], shape=(3, (3, 2, 3)))

        >>> zeros_like(JaggedArray(np.arange(22), (4, 2, (1, 2, 1, 2), (3, 1, 4, 1))))
        JaggedArray([0, 0, ...], shape=(4, 2, (1, 2, 1, 2), (3, 1, 4, 1)))
    """
    return zeros(arr.shape, dtype=dtype)


def ones(shape: JaggedShapeLike, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create a jagged array of ones in a given jagged shape.

    Args:
        shape:
            the shape of the array to produce.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> ones((3, (3, 2, 3)))
        JaggedArray([1, 1, 1,...], shape=(3, (3, 2, 3)))

        >>> ones((2, 2, (1, 2)), dtype=bool)
        JaggedArray([True, True, ...], shape=(2, 2, (1, 2)))

     """
    raise NotImplementedError


def ones_like(arr: JaggedArray, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create a jagged array of ones in the shape of another jagged array.

    Args:
        arr:
            the jagged array to use as template.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> ones_like(JaggedArray(np.arange(8), (3, (3, 2, 3))))
        JaggedArray([1, 1, ...], shape=(3, (3, 2, 3)))

        >>> ones_like(JaggedArray(np.arange(22), (4, 2, (1, 2, 1, 2), (3, 1, 4, 1))))
        JaggedArray([1, 1, ...], shape=(4, 2, (1, 2, 1, 2), (3, 1, 4, 1)))
    """
    return ones(arr.shape, dtype=dtype)


def full(shape: JaggedShapeLike, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create a jagged array of a given value in a given jagged shape.

    Args:
        shape:
            the shape of the array to produce.

        dtype:
            the dtype of the array to produce.

        fill_value:
            the value with which to fill the array.

    Examples:
        >>> ones((3, (3, 2, 3)))
        JaggedArray([1, 1, 1,...], shape=(3, (3, 2, 3)))

        >>> ones((2, 2, (1, 2)), dtype=bool)
        JaggedArray([True, True, ...], shape=(2, 2, (1, 2)))

     """
    raise NotImplementedError


def full_like(
    arr: JaggedArray, fill_value, dtype: Optional[DtypeLike] = None
) -> JaggedArray:
    """ Create a jagged array of a given value in the shape of another jagged array.

    Args:
        arr:
            the jagged array to use as template.

        fill_value:
            the value with which to fill the array.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> ones_like(JaggedArray(np.arange(8), (3, (3, 2, 3))))
        JaggedArray([1, 1, ...], shape=(3, (3, 2, 3)))

        >>> ones_like(JaggedArray(np.arange(22), (4, 2, (1, 2, 1, 2), (3, 1, 4, 1))))
        JaggedArray([1, 1, ...], shape=(4, 2, (1, 2, 1, 2), (3, 1, 4, 1)))
    """
    return ones(arr.shape, dtype=dtype)


def array_equal(x: JaggedArray, y: JaggedArray) -> bool:
    """ Evaluate whether two jagged arrays are equal.

    Examples:
        >>> array_equal(
        ...    JaggedArray(np.arange(8), (3, (3, 2, 3))),
        ...    JaggedArray(np.arange(8), shapes=[[3], [2], [3]])
        ... )
        True

        With same shapes, but different data:
        >>> array_equal(
        ...    JaggedArray(np.arange(8), (3, (3, 2, 3))),
        ...    JaggedArray(np.arange(8, 0, -1), (3, (3, 2, 3))),
        ... )
        False

        With same data, but different shapes:
        >>> array_equal(
        ...    JaggedArray(np.arange(8), (3, (3, 2, 3))),
        ...    JaggedArray(np.arange(8), (3, (3, 3, 2))),
        ... )
        False
    """
    return np.array_equal(x.data, y.data) and x.shape == y.shape


def random(
    shape: ArrayLike,
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
        JaggedArray([0.73199394 0.59865848 0.15601864], shape=(2, (1, 2), (1, 1)))

        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> jagged.random((3, 3), random_state=rng, data_rvs=lambda n: rng.randint(0, 10, n))
        JaggedArray([6, 3, 7, ... 8, 2, 4], shape=(3, (3, 4), 1, (4, 5)))
    """
    raise NotImplementedError


def where(condition: JaggedArray, x: JaggedArray, y: JaggedArray):
    """ Return elements chosen from between two arrays depending on a condition.

    Args:
        condition:
            The condition
        x, y:
            The arrays from which to choose values

    Examples:
        >>> import jagged
        >>> jagged.where(
        ...     JaggedArray([True, False, True, False, True], shape=(3, (2, 1, 2))),
        ...     JaggedArray(np.arange(5), shape=(3, (2, 1, 2))),
        ...     JaggedArray(-np.arange(5), shape=(3, (2, 1, 2))),
        ... )
        JaggedArray([0, -1, 2, -3, 4], shape=(3, (2, 1, 2)))
     """
    raise NotImplementedError


def squeeze(arr: JaggedArray, axis: AxisLike) -> JaggedArray:
    """ Squeeze the axes of a jagged array.

    This removes single dimensional axes from the jagged array.

    Args:
        axis:
            the axes to squeeze.

    Examples:
        >>> squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3))))
        JaggedArray([0, 1, 2...], shape=(3, (3, 2, 3)))

        Squeezing multiple axes at once:

        >>> squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3), 1))
        JaggedArray([0, 1, 2...], shape=(3, (3, 2, 3)))

        Squeezing a particular axis:

        >>> squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3), 1)), axis=1)
        JaggedArray([0, 1, 2...], shape=(3, (3, 2, 3), 1))

        Squeezing multiple particular axes:

        >>> squeeze(JaggedArray(np.arange(7), (3, 1, 1, (3, 2, 3), 1)), axis=(1, 2))
        JaggedArray([0, 1, 2...], shape=(3, (3, 2, 3), 1))

        Trying to squeeze an axis with more than one entry:

        >>> jagged.squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3))), axis=2)
        Traceback (most recent call last):
            ...
        ValueError: cannot select an axis to squeeze out which has size not equal to one

        Trying to squeeze the inducing axis:

        >>> jagged.squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3))), axis=0)
        Traceback (most recent call last):
            ...
        ValueError: cannot select an axis to squeeze out which has size not equal to one

        Squeezing the inducing axis when it is only of length one:

        >>> jagged.squeeze(JaggedArray(np.arange(4), (1, 2, 2)), axis=0)
        array([[0, 1],
               [2, 3]])

    See Also:
        JaggedArray.squeeze: equivalent function as jagged array method
    """
    raise NotImplementedError


def expand_dims(arr: JaggedArray, axis: int = -1) -> JaggedArray:
    """ Add a dimension.

    Args:
        arr:
            The jagged array which to add the dimension.
        axis:
            The axis after which to add the dimension.

    Examples:
        >>> import jagged
        >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
        >>> jagged.expand_dims(ja, axis=1)
        JaggedArray([0, 1, 2, ...], shape=(3, 1, (3, 2, 3))

        >>> jagged.expand_dims(ja, axis=-1)
        JaggedArray([0, 1, 2, ...], shape=(3, (3, 2, 3), 1)

    See Also:
        JaggedArray.expand_dims: equivalent function as jagged array method
    """
    raise NotImplementedError


def concatenate(objs: Iterable[JaggedArray], axis: int = 0) -> JaggedArray:
    """ Concatenate data along axes for jagged arrays.

    Args:
        objs:
            The jagged arrays to concatenate.

        axis:
            The axis along which to concatenate.

    Examples:
        >>> ja = JaggedArray([0, 1, 2,...], shape=(3, (3, 2, 3), (3, 6, 4)))

        >>> jagged.concatenate([ja, ja], axis=0)
        JaggedArray([0, 1, 2,...], shape=(6, (3, 2, 3, 3, 2, 3), (3, 6, 4, 3, 6, 4)))

        >>> jagged.concatenate([ja, ja], axis=1)
        JaggedArray([0, 1, 2,...], shape=(3, (6, 4, 6), (3, 6, 4)))

        >>> jagged.concatenate([ja, ja], axis=2)
        JaggedArray([0, 1, 2,...], shape=(3, (3, 2, 4), (6, 12, 8)))

        >>> jagged.concatenate([ja, ja], axis=-1)
        JaggedArray([0, 1, 2,...], shape=(3, (3, 2, 4), (6, 12, 8)))
    """
    raise NotImplementedError


def stack(objs: Iterable[JaggedArray], axis: Optional[int] = -1) -> JaggedArray:
    """ Stack JaggedArrays on a new axis.

    Args:
        objs:
            The jagged arrays to stack.

        axis:
            The axis in the result array along which the arrays are stacked.

    Notes:
        It is not possible to stack along the 0'th axis, as this is the jagged
        inducing dimension.

    Examples:
        >>> ja = JaggedArray(np.arange(33), (3, (3, 2, 3), (3, 6, 4))

        >>> jagged.stack([ja, ja])
        JaggedArray([0, 1, 2, ...], shape=(3, (3, 2, 3), (3, 6, 4), 2))

        >>> jagged.stack([ja, ja], axis=1)
        JaggedArray([0, 1, 2, ...], shape=(3, (3, 2, 3), (3, 6, 4), 2))

        >>> jagged.stack([ja, ja], axis=0)
        Traceback (most recent call last):
            ...
        ValueError: cannot stack over the jagged inducing dimension

    """
    raise NotImplementedError


def diagonal(arr: JaggedArray, offset: int = 0, axis1: int = 0, axis2: int = 1):
    """ Return specified diagonals.

    Args:
        arr:
            The jagged array from which to fetch the diagonals

        offset:
            Offset of the diagonal from the main diagonal. Can be both positive and
            negative to access upper and lower triangle respectively.

        axis1, axis2:
            Axes to be used as the first and second axis of the subarrays from
            which the diagonals should be taken.

    See also:
        JaggedArray.diagonal: equivalent function as method on jagged array
        numpy.diagonal: equivalent function in numpy
    """
    raise NotImplementedError


def trace(
    arr: JaggedArray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: Optional[DtypeLike] = None,
    out: Optional[np.ndarray] = None,
):
    """ Return the sum along diagonals of a jagged array.

    Args:
        arr:
            The jagged array from which to fetch the diagonals

        offset:
            Offset of the diagonal from the main diagonal. Can be both positive and
            negative to access upper and lower triangle respectively.

        axis1, axis2:
            Axes to be used as the first and second axis of the subarrays from
            which the diagonals should be taken.

        dtype:
            The data-type of the returned array.

        out:
            The array in which the output is placed.

    See also:
        JaggedArray.trace: equivalent function as method on jagged array
        numpy.trace: equivalent function in numpy
    """
    raise NotImplementedError
