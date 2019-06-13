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
from collections.abc import Iterable
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from .typing import AxisLike
from .typing import JaggedMetadata
from .typing import JaggedShapeLike


def is_float(obj: Any) -> bool:
    """ Whether an object is a float.

    Args:
        obj:
            the object to test

    Examples:
        >>> is_float(0.1)
        True

        >>> is_float(np.float32(0.1))
        True

        >>> is_float(1)
        False

        >>> is_float(np.int64(1))
        False

        >>> is_float(None)
        False
    """
    return isinstance(obj, (float, np.floating))


def is_integer(obj: Any) -> bool:
    """ Whether an object is an integer.

    Args:
        obj:
            the object to test

    Examples:
        >>> is_integer(1)
        True

        >>> is_integer(np.int32(1))
        True

        >>> is_integer(np.int16(1))
        True

        >>> is_integer(0.1)
        False

        >>> is_integer(None)
        False
    """
    return isinstance(obj, (int, np.integer))


def is_iterable(obj):
    """ Whether an object is iterable.

    Args:
        obj:
            the object to test.

    Notes:
        For the purpose of this function, a string is NOT iterable.

    Examples:
        >>> is_iterable([1, 2])
        True

        >>> is_iterable(np.array([1, 2]))
        True

        >>> is_iterable((1, 2))
        True

        >>> is_iterable(1)
        False

        >>> is_iterable('not iterable')
        False

        >>> is_iterable(None)
        False
    """
    if isinstance(obj, str):
        return False
    else:
        return isinstance(obj, Iterable)


def metadata_to_array(inducing: int, metadata: JaggedMetadata) -> np.ndarray:
    """ Turn jagged metadata into an array

    Examples:
        >>> metadata_to_array(3, ((1, 3, 2),))
        array([[1],
               [3],
               [2]])

        >>> metadata_to_array(3, ((1, 3, 2), (3, 2, 3)))
        array([[1, 3],
               [3, 2],
               [2, 3]])

        >>> metadata_to_array(3, ((3, 1, 1), 2))
        array([[3, 2],
               [1, 2],
               [1, 2]])
    """

    if any(len(ax) != inducing for ax in metadata if isinstance(ax, tuple)):
        msg = "Jagged axes must have number of entries equal to length of inducing dim"
        raise ValueError(msg)
    res = np.empty((inducing, len(metadata)), dtype=int)
    for i, ax in enumerate(metadata):
        res[:, i] = ax
    return res


def array_to_metadata(arr: np.ndarray) -> JaggedMetadata:
    """ Turn an array into metadata

    Examples:
        >>>array_to_metadata(np.array([[1], [3], [2]]))
        (3, (1, 3, 2))

        >>> metadata_to_array(np.array([[1, 3], [3, 2], [2, 3]]))
        (3, (1, 3, 2), (3, 2, 3))

        >>> metadata_to_array(np.array([[3, 2], [1, 2], [1, 2]]))
        (3, (3, 1, 1), 2)
    """

    return tuple(
        dim
        if is_integer(dim)
        else dim[0]
        if all(jdim == dim[0] for jdim in dim)
        else tuple(dim)
        for dim in arr.T
    )


def sanitize_shape(shape):
    """ Canonicalize a jagged shape

    Args:
        shape:
            The shape to canonicalize

    Examples:
        >>> sanitize_shape((3, (1, 2, 3)))
        (3, (1, 2, 3))

        >>> sanitize_shape([3, [1, 2, 3]])
        (3, (1, 2, 3))

        >>> sanitize_shape((3, (1, 1, 1), (1, 2, 3)))
        (3, 1, (1, 2, 3))

        >>> sanitize_shape((3, (1, 2)))
        Traceback (most recent call last):
        ...
        ValueError: Shape for jagged axes must have entries equal to length of inducing axis.
    """

    n_inducing = shape[0]
    res = [n_inducing]

    for i, dim in enumerate(shape[1:]):
        if is_iterable(dim):
            if len(dim) != n_inducing:
                msg = "Shape for jagged axes must have entries equal to length of inducing axis."
                raise ValueError(msg)
            elif all(dim[0] == d for d in dim):
                res.append(int(dim[0]))
            else:
                if isinstance(dim, np.ndarray):
                    dim = tuple(dim.tolist())
                else:
                    dim = tuple(int(d) for d in dim)
                res.append(dim)
        else:
            res.append(int(dim))
    return tuple(res)


def shape_to_shapes(shape: JaggedShapeLike) -> np.ndarray:
    """ Convert a jagged shape to shapes.

    Args:
        shape:
            A jagged shape

    Examples:
        >>> shape_to_shapes((3, (1, 3, 2)))
        array([[1],
               [3],
               [2]])

        >>> shape_to_shapes((3, (1, 3, 2), (3, 2, 3)))
        array([[1, 3],
               [3, 2],
               [2, 3]])

        >>> shape_to_shapes((3, (3, 1, 1), 2))
        array([[3, 2],
               [1, 2],
               [1, 2]])

        >>> shape_to_shapes((4, (1, 3, 2)))
        Traceback (most recent call last):
            ...
        ValueError: invalid shape. Jagged axes must have number of entries equal to length of inducing dim

    See Also:
        shapes_to_shape: the reverse of this function
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
    if not np.issubdtype(shapes.dtype, np.signedinteger):
        raise ValueError("Shapes must be integers.")
    return sanitize_shape((len(shapes), *shapes.T))


def shapes_to_size(shapes: JaggedShapeLike) -> int:
    """ Get the number of entries for an array of shapes.

    Args:
        shapes:
            An array of shapes

    Examples:
        >>> shapes_to_size(np.array([[1], [2]]))
        3

        >>> shapes_to_size(np.array([[2, 2], [3, 2], [2, 2]]))
        14

        >>> shapes_to_size(np.array([[2, 1], [3, 4], [2, 2]]))
        18

        >>> shapes_to_size(np.array([[2, 2, 1], [2, 3, 4], [2, 2, 2]]))
        36
    """
    return shapes.prod(axis=1).sum()


def shape_to_size(shape: JaggedShapeLike) -> int:
    """ Get the number of entries for an array of shapes.

    Args:
        shape:
            A jagged shape

    Examples:
        >>> shape_to_size((2, (1, 2)))
        3

        >>> shape_to_size((3, (2, 3, 2), 2))
        14

        >>> shape_to_size((3, (2, 3, 2), (1, 4, 2)))
        18

        >>> shape_to_size((3, 2, (2, 3, 2), (1, 4, 2)))
        36
    """

    return shapes_to_size(shape_to_shapes(shape))


def shape_is_jagged(shape: JaggedShapeLike):
    """ whether a shape is jagged.

    Args:
        shape:
            the shape to check

    >>> shape_is_jagged((3, (3, 2, 3)))
    True

    >>> shape_is_jagged((3, 3))
    False

    >>> shape_is_jagged((3, (3, 3, 3)))
    False
    """
    return any(is_iterable(i) for i in sanitize_shape(shape))


def jagged_to_string(
    jarr,
    max_line_width: Optional[int] = None,
    precision: Optional[int] = None,
    suppress_small: Optional[bool] = None,
    separator: Optional[str] = "",
    prefix: Optional[str] = "",
    suffix: Optional[str] = "",
    formatter: Optional[Dict[str, callable]] = None,
    threshold: Optional[int] = None,
    edgeitems: Optional[int] = None,
    sign: Optional[str] = None,
    floatmode: Optional[str] = None,
    legacy: Optional[Union[str, bool]] = None,
):
    """ Return a string representation of a jagged array.

    Args:
        see `numpy.array2string` for full documentation.
    """
    # create a custom formatter for all the data
    formatter = formatter or {
        "all": np.core.arrayprint._get_format_function(
            jarr.flatten(), **np.get_printoptions()
        )
    }
    delim = separator + "\n" * (len(jarr.shape) - 1)
    middle = delim.join(
        (" " * len(prefix) if i > 0 else "")
        + np.array2string(
            arr,
            max_line_width=max_line_width,
            precision=precision,
            suppress_small=suppress_small,
            prefix=prefix,
            suffix=suffix,
            separator=separator,
            formatter=formatter,
            sign=sign,
            floatmode=floatmode,
            legacy=legacy,
        )
        for i, arr in enumerate(jarr)
    )
    return prefix + middle + suffix


def sanitize_axis(axis: AxisLike, ndim: int, multi=True) -> Optional[Tuple[int]]:
    """ Coerce multiple axes input into canonical format.

    The canonical format is a tuple of positive integers specifying the axes.

    Args:
        axis:
            the axes input to coerce into a canonical format.

        ndim:
            the total number of dimensions.

        multi:
            whether multiple axes may be passed.

    Examples:

        >>> sanitize_axis(1, 3)
        1

        >>> sanitize_axis((1, 2), 4)
        (1, 2)

        >>> sanitize_axis(-1, 3)
        2

        >>> sanitize_axis((0, -1), 3)
        (0, 2)

        >>> sanitize_axis([1, 2], 4)
        (1, 2)

        >>> sanitize_axis(np.array([1, 2]), 4)
        (1, 2)

        >>> sanitize_axis((0, 1), 2, multi=False)
        Traceback (most recent call last):
            ...
        ValueEorror: multiple axes are not allowed

        >>> sanitize_axis(4, 3)
        Traceback (most recent call last):
            ...
        ValueError: axis 4 is out of bounds

        >>> sanitize_axis(-4, 3)
        Traceback (most recent call last):
            ...
        ValueError: axis -4 is out of bounds

        >>> sanitize_axis((3, 3), 4)
        Traceback (most recent call last):
            ...
        ValueError: duplicate value in 'axis'

        >>> sanitize_axis((3, -1), 4)
        Traceback (most recent call last):
            ...
        ValueError: duplicate value in 'axis'

        >>> sanitize_axis('one', 4)
        Traceback (most recent call last):
            ...
        ValueError: 'str' object cannot be interpreted as an integer
    """

    if axis is None:
        pass
    elif is_integer(axis):
        axis = int(axis)
        if axis < 0:
            axis += ndim
        if not (0 <= axis < ndim):
            msg = f"axis {axis} is out of bounds"
            raise ValueError(msg)
    elif is_iterable(axis):
        if not multi:
            raise ValueError("multiple axes are not allowed")
        axis = tuple(sanitize_axis(ax, ndim) for ax in axis)
        if len(set(axis)) < len(axis):
            raise ValueError("duplicate value in 'axis'")
    else:
        raise ValueError(f"'{type(axis)} object cannot be interpreted as an integer")

    return axis


if __name__ == "__main__":
    import doctest

    doctest.testmod()
