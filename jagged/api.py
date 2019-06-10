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
from functools import partial
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np

from .core import JaggedArray
from .factories import ascontiguousarray
from .typing import ArrayLike
from .typing import AxisLike
from .typing import DtypeLike
from .typing import JaggedShapeLike
from .typing import Number
from .typing import RandomState
from .typing import ShapeLike
from .utils import is_integer
from .utils import sanitize_axis
from .utils import shape_is_jagged
from .utils import shape_to_size


def arange(
    *args,
    start: Number = None,
    stop: Number = None,
    step: Number = None,
    shape: ShapeLike = None,
    dtype: DtypeLike = None,
):
    """ interval reshaped to a given shape

    Args:
        args:
            Positional args as per np.arange: [start,] stop,[ step,]
            These will be overwritten with provided kwargs

        start:
            the start of interval (inclusive) - default is 0

        stop:
            the end of the interval (exclusive)

        step:
            the spacing between values

        dtype:
            the type of output array

    Examples:
        >>> import jagged
        >>> jagged.arange(shape=(3, (3, 2, 3)))
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        >>> jagged.arange(5, 13, shape=(3, (3, 2, 3)))
        JaggedArray([[ 5,  6,  7],
                     [ 8,  9],
                     [10, 11, 12]])

        >>> jagged.arange(0, 15, 2, shape=(3, (3, 2, 3)))
        JaggedArray([[ 0,  2,  4],
                     [ 6,  8],
                     [10, 12, 14]])

    See Also:
        numpy.arange
    """
    if shape is None and stop is None:
        raise ValueError("arange requires either a `shape` or `stop`.")

    dtype = np.int if dtype is None else dtype

    if len(args) == 1:
        stop, = args
    elif len(args) == 2:
        start, stop = args
    elif len(args) == 3:
        start, stop, step = args

    start = start or 0
    step = step or 1

    if stop is not None:
        size = int(np.ceil((stop - start) / step))

    if shape is None:
        shape = (size,)
    else:
        shape_size = shape_to_size(shape)
        if stop is not None and size != shape_size:
            raise ValueError(f"range with size {size} cannot have given shape {shape}")
        stop = step * (start + shape_size)

    return JaggedArray(shape, buffer=np.arange(start, stop, step), dtype=dtype)


def zeros(shape: JaggedShapeLike, dtype: Optional[DtypeLike] = None):
    """ Create a jagged array of zeros in a given jagged shape.

    Args:
        shape:
            the shape of the array to produce

        dtype:
            the dtype of the array to produce

    Examples:
        >>> import jagged
        >>> jagged.zeros((3, (3, 2, 3)))
        JaggedArray([[0., 0., 0.],
                     [0., 0.],
                     [0., 0., 0.]])

        >>> jagged.zeros((2, (2, 1)), dtype=bool)
        JaggedArray([[False, False],
                     [False]], dtype=bool)
     """
    return JaggedArray(np.zeros(shape_to_size(shape), dtype), shape)


def zeros_like(arr: JaggedArray, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create a jagged array of zeros in the shape of another jagged array.

    Args:
        arr:
            the jagged array to use as template.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray
        >>> jagged.zeros_like(JaggedArray(np.arange(8), (3, (3, 2, 3))))
        JaggedArray([[0, 0, 0],
                     [0, 0],
                     [0, 0, 0]])

        >>> jagged.zeros_like(JaggedArray(np.arange(12), (4, 2, (1, 2, 1, 2))))
        JaggedArray([[[0],
                      [0]],
        <BLANKLINE>
                     [[0, 0],
                      [0, 0]],
        <BLANKLINE>
                     [[0],
                      [0]],
        <BLANKLINE>
                     [[0, 0],
                      [0, 0]]])
    """
    return zeros(arr.shape, dtype=dtype or arr.dtype)


def ones(shape: JaggedShapeLike, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create a jagged array of ones in a given jagged shape.

    Args:
        shape:
            the shape of the array to produce.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> import jagged
        >>> jagged.ones((3, (3, 2, 3)))
        JaggedArray([[1., 1., 1.],
                     [1., 1.],
                     [1., 1., 1.]])

        >>> jagged.ones((2, (2, 1)), dtype=bool)
        JaggedArray([[ True,  True],
                     [ True]], dtype=bool)
     """
    return JaggedArray(np.ones(shape_to_size(shape), dtype), shape)


def ones_like(arr: JaggedArray, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create a jagged array of ones in the shape of another jagged array.

    Args:
        arr:
            the jagged array to use as template.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray

        >>> jagged.ones_like(JaggedArray(np.arange(8), (3, (3, 2, 3))))
        JaggedArray([[1, 1, 1],
                     [1, 1],
                     [1, 1, 1]])

        >>> jagged.ones_like(JaggedArray(np.arange(12), (4, 2, (1, 2, 1, 2))))
        JaggedArray([[[1],
                      [1]],
        <BLANKLINE>
                     [[1, 1],
                      [1, 1]],
        <BLANKLINE>
                     [[1],
                      [1]],
        <BLANKLINE>
                     [[1, 1],
                      [1, 1]]])
    """
    return ones(arr.shape, dtype=dtype or arr.dtype)


def full(
    shape: JaggedShapeLike, fill_value, dtype: Optional[DtypeLike] = None
) -> JaggedArray:
    """ Create a jagged array of a given value in a given jagged shape.

    Args:
        shape:
            the shape of the array to produce.

        fill_value:
            the value with which to fill the array.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray

        >>> jagged.full((3, (3, 2, 3)), 42)
        JaggedArray([[42, 42, 42],
                     [42, 42],
                     [42, 42, 42]])

        >>> jagged.full((2, (2, 1)), 3.14, dtype='f4')
        JaggedArray([[3.14, 3.14],
                     [3.14]], dtype=float32)
     """
    return JaggedArray(np.full(shape_to_size(shape), fill_value, dtype), shape)


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
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray

        >>> jagged.full_like(JaggedArray(np.arange(8), (3, (3, 2, 3))), 42)
        JaggedArray([[42, 42, 42],
                     [42, 42],
                     [42, 42, 42]])

        >>> jagged.full_like(JaggedArray(np.arange(12), (4, 2, (1, 2, 1, 2))), 42)
        JaggedArray([[[42],
                      [42]],
        <BLANKLINE>
                     [[42, 42],
                      [42, 42]],
        <BLANKLINE>
                     [[42],
                      [42]],
        <BLANKLINE>
                     [[42, 42],
                      [42, 42]]])
    """
    return full(arr.shape, fill_value, dtype=dtype or arr.dtype)


def empty(shape: JaggedShapeLike, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create an empty jagged array in a given jagged shape.

    Args:
        shape:
            the shape of the array to produce.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> import jagged

        >>> jagged.empty((3, (3, 2, 3)))  # doctest:+SKIP
        JaggedArray([[0., 0., 0.],
                     [0., 0.],
                     [0., 0., 0.]])

        >>> jagged.empty((2, 2, (1, 2)), dtype=bool)  # doctest:+SKIP
        JaggedArray([[False, False],
                     [False]])
     """
    return JaggedArray(np.empty(shape_to_size(shape), dtype), shape)


def empty_like(arr: JaggedArray, dtype: Optional[DtypeLike] = None) -> JaggedArray:
    """ Create an empty jagged array in the shape of another jagged array.

    Args:
        arr:
            the jagged array to use as template.

        dtype:
            the dtype of the array to produce.

    Examples:
        >>> import numpy as np
        >>> from jagged import JaggedArray
        >>> import jagged

        >>> jagged.empty_like(JaggedArray(np.arange(8), (3, (3, 2, 3))))  # doctest:+SKIP
        JaggedArray([[0, 0, 0],
                     [0, 0],
                     [0, 0, 0]])

        >>> jagged.empty_like(JaggedArray(np.arange(12), (4, 2, (1, 2, 1, 2))))  # doctest:+SKIP
        JaggedArray([[[0],
                      [0]],
        <BLANKLINE>
                     [[0, 0],
                      [0, 0]],
        <BLANKLINE>
                     [[0],
                      [0]],
        <BLANKLINE>
                     [[0, 0],
                      [0, 0]]])
    """
    return empty(arr.shape, dtype=dtype or arr.dtype)


def array_equal(x: JaggedArray, y: JaggedArray) -> bool:
    """ Evaluate whether two jagged arrays are equal.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray

        >>> jagged.array_equal(
        ...    JaggedArray(np.arange(8), (3, (3, 2, 3))),
        ...    JaggedArray(np.arange(8), shapes=[[3], [2], [3]])
        ... )
        True

        With same shapes, but different data:
        >>> jagged.array_equal(
        ...    JaggedArray(np.arange(8), (3, (3, 2, 3))),
        ...    JaggedArray(np.arange(8, 0, -1), (3, (3, 2, 3))),
        ... )
        False

        With same data, but different shapes:
        >>> jagged.array_equal(
        ...    JaggedArray(np.arange(8), (3, (3, 2, 3))),
        ...    JaggedArray(np.arange(8), (3, (3, 3, 2))),
        ... )
        False
    """
    return np.array_equal(x.data, y.data) and x.shape == y.shape


def allclose(
    x: JaggedArray,
    y: JaggedArray,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
):
    """ Whether two jagged arrays are element-wise equal withing a given tolerance.

    Args:
        jarr{1,2}:
            The arrays between which to test the closeness

        rtol:
            The relative tolerance

        atol:
            The absolute tolerance

        equal_nan:
            Whether to compare NaNs as equal
    """
    return np.allclose(x.data, y.data)


def random(
    shape: ArrayLike,
    jagged_axes: Optional[Tuple[int]] = None,
    allow_empty: bool = False,
    random_state: Optional[RandomState] = None,
    data_rvs: Optional[Callable] = None,
    dtype: DtypeLike = None,
):
    """ Generate a random jagged array.

    Args:
        shape:
            If a jagged shape, the shape of the resulting jagged array.
            If a flat shape, the maximal bounds of the jagged array.
        jagged_axes:
            The indices of the axes that are to be jagged.
        allow_empty:
            Whether to allow empty subarrays (i.e. with dim = 0)
        random_state:
            rng or random seed. If not given, `np.random` will be used.
        data_rvs:
            Data generation callback
        dtype:
            The dtype to use.

    Examples:
        >>> import jagged
        >>> jagged.random((2, 2, 2), random_state=42)
        JaggedArray([[[0.95071431],
                      [0.73199394]],
        <BLANKLINE>
                     [[0.59865848, 0.15601864],
                      [0.15599452, 0.05808361]]])

        With a custom data_rvs:

        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> jagged.random((3, 3), random_state=rng, data_rvs=lambda n: rng.randint(0, 10, n))
        JaggedArray([[7],
                     [4, 6],
                     [9]])

        With a jagged shape:

        >>> jagged.random((3, (3, 2, 3)), random_state=42)
        JaggedArray([[0.37454012, 0.95071431, 0.73199394],
                     [0.59865848, 0.15601864],
                     [0.15599452, 0.05808361, 0.86617615]])

        With a particular dtype (ints sample from all possible values):

        >>> jagged.random((3, 2), random_state=42, dtype=np.uint8)
        JaggedArray([[179],
                     [ 61],
                     [234, 203]], dtype=uint8)
    """

    if random_state is None:
        random_state = np.random
    elif is_integer(random_state):
        random_state = np.random.RandomState(random_state)

    if data_rvs is None:
        if np.issubdtype(dtype, np.integer):
            data_rvs = partial(
                random_state.randint,
                np.iinfo(dtype).min,
                np.iinfo(dtype).max,
                dtype=dtype,
            )
        elif np.issubdtype(dtype, np.complexfloating):

            def data_rvs(n):
                return random_state.rand(n) + random_state.rand(n) * 1j

        else:
            data_rvs = random_state.rand

    if not shape_is_jagged(shape):
        limits, shape = shape, list(shape)
        # we must choose the shape
        if limits[0] <= 1:
            msg = "Cannot create a jagged array with inducing axis of dimension 1."
            raise ValueError(msg)
        if np.prod(limits) <= 2:
            raise ValueError("Cannot create a jagged array with two or less entries.")
        if jagged_axes is None:
            # we must choose the jagged axes
            # there must be at least one, and they can't have limits of dimension 1
            possibly_jagged = [i for i, ax in enumerate(limits) if ax != 1 and i != 0]
            if len(possibly_jagged) > 1:
                n_to_choose = random_state.randint(1, len(possibly_jagged))
            else:
                n_to_choose = 1

            jagged_axes = random_state.choice(
                possibly_jagged, n_to_choose, replace=False
            )
        else:
            if any(limits[ja] == 1 for ja in jagged_axes):
                raise ValueError("Jagged axes cannot have dimension 1")
        for ax in jagged_axes:
            # come up with the jagged shape
            lower = 0 if allow_empty else 1
            dim = random_state.randint(lower, 1 + limits[ax], limits[0]).tolist()
            if all(d == dim[0] for d in dim):
                # if all the same value, randomly perturb one value
                diff = -1 if dim[0] > 1 else +1
                dim[random_state.randint(0, limits[0])] += diff
            shape[ax] = dim
    return JaggedArray(shape, buffer=data_rvs(shape.size))


def where(condition: JaggedArray, x: JaggedArray, y: JaggedArray):
    """ Return elements chosen from between two arrays depending on a condition.

    Args:
        condition:
            The condition
        x, y:
            The arrays from which to choose values

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray
        >>> jagged.where(
        ...     JaggedArray([True, False, True, False, True], shape=(3, (2, 1, 2))),
        ...     JaggedArray(np.arange(5), shape=(3, (2, 1, 2))),
        ...     JaggedArray(-np.arange(5), shape=(3, (2, 1, 2))),
        ... )
        JaggedArray([[ 0, -1],
                     [ 2],
                     [-3,  4]])
    """
    # TODO: broadcasting
    if condition.shape != x.shape != y.shape:
        raise ValueError("`where` does not yet operate on differently shaped arrays.")
    return JaggedArray(np.where(condition.data, x.data, y.data), condition.shape)


def squeeze(jarr: JaggedArray, axis: Optional[AxisLike] = None) -> JaggedArray:
    """ Squeeze the axes of a jagged array.

    This removes single dimensional axes from the jagged array.

    Args:
        jarr:
            the jagged array to squeeze.
        axis:
            the axes of the array to squeeze.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray
        >>> jagged.squeeze(JaggedArray(np.arange(8), (3, 1, (3, 2, 3))))
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        Squeezing multiple axes at once:

        >>> jagged.squeeze(JaggedArray(np.arange(8), (3, 1, (3, 2, 3), 1)))
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        Squeezing a particular axis:

        >>> jagged.squeeze(JaggedArray(np.arange(8), (3, 1, (3, 2, 3), 1)), axis=-1)
        JaggedArray([[[0, 1, 2]],
        <BLANKLINE>
                     [[3, 4]],
        <BLANKLINE>
                     [[5, 6, 7]]])

        >>> _.shape
        (3, 1, (3, 2, 3))

        Squeezing multiple particular axes:

        >>> jagged.squeeze(JaggedArray(np.arange(8), (3, 1, 1, (3, 2, 3), 1)), axis=(1, 2))
        JaggedArray([[[0],
                      [1],
                      [2]],
        <BLANKLINE>
                     [[3],
                      [4]],
        <BLANKLINE>
                     [[5],
                      [6],
                      [7]]])

        >>> _.shape
        (3, (3, 2, 3), 1)

        Trying to squeeze an axis with more than one entry:

        >>> jagged.squeeze(JaggedArray(np.arange(8), (3, 1, (3, 2, 3))), axis=2)
        Traceback (most recent call last):
            ...
        ValueError: cannot select an axis to squeeze out which has size not equal to one

        Trying to squeeze the inducing axis:

        >>> jagged.squeeze(JaggedArray(np.arange(8), (3, 1, (3, 2, 3))), axis=0)
        Traceback (most recent call last):
            ...
        ValueError: cannot select an axis to squeeze out which has size not equal to one

        Squeezing the inducing axis when it is only of length one:

        >>> import warnings
        >>> with warnings.catch_warnings():
        ...    warnings.simplefilter("ignore")
        ...    ja = JaggedArray(np.arange(4), (1, 2, 2))
        >>> jagged.squeeze(ja, axis=0)
        array([[0, 1],
               [2, 3]])

    See Also:
        JaggedArray.squeeze: equivalent function as jagged array method
    """
    axis = sanitize_axis(axis, jarr.ndim)
    if is_integer(axis):
        axis = (axis,)

    if axis is None or 0 in axis:
        # we should try to squeeze the inducing axis
        if jarr.shape[0] == 1:
            # special case that inducing axis can be squeezed
            if axis not in (None, (0,)):
                # we have to adapt the axis down
                axis = tuple(ax - 1 for ax in axis if ax != 0)
            else:
                axis = None
            return np.squeeze(jarr[0], axis)

    if axis is None:
        axis = list(range(jarr.ndim))
    else:
        for ax in axis:
            if ax in jarr.shape.jagged_axes or jarr.shape[ax] > 1:
                msg = "cannot select an axis to squeeze out which has size not equal to one"
                raise ValueError(msg)

    shape = tuple(
        dim
        for i, dim in enumerate(jarr.shape)
        if i in jarr.jagged_axes or i not in axis or dim > 1
    )
    return JaggedArray(shape, buffer=jarr.data)


def expand_dims(jarr: JaggedArray, axis: int = -1) -> JaggedArray:
    """ Add a dimension.

    Args:
        jarr:
            The jagged array which to add the dimension.
        axis:
            The axis after which to add the dimension.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray

        >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
        >>> jagged.expand_dims(ja, axis=1)
        JaggedArray([[[0, 1, 2]],
        <BLANKLINE>
                     [[3, 4]],
        <BLANKLINE>
                     [[5, 6, 7]]])

        >>> jagged.expand_dims(ja, axis=-1)
        JaggedArray([[[0],
                      [1],
                      [2]],
        <BLANKLINE>
                     [[3],
                      [4]],
        <BLANKLINE>
                     [[5],
                      [6],
                      [7]]])

        >>> jagged.expand_dims(ja, axis=0)
        Traceback (most recent call last):
            ...
        ValueError: cannot expand before the jagged inducing dimension

    See Also:
        JaggedArray.expand_dims: equivalent function as jagged array method
    """
    axis = sanitize_axis(axis, jarr.ndim + 1, multi=False)
    shape = jarr.shape
    if axis == 0:
        raise ValueError("cannot expand before the jagged inducing dimension")
    else:
        axis = axis if axis >= 0 else len(shape) - axis
        shape = (*shape[:axis], 1, *shape[axis:])
        return jarr.reshape(shape)


def concatenate(jarrs: Iterable[JaggedArray], axis: int = 0) -> JaggedArray:
    """ Concatenate data along axes for jagged arrays.

    Args:
        jarrs:
            The jagged arrays to concatenate.

        axis:
            The axis along which to concatenate.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray
        >>> ja1 = JaggedArray(np.arange(8), shape=(3, (3, 2, 3)))
        >>> ja2 = JaggedArray(np.arange(9), shape=(3, (4, 3, 2)))
        >>> jagged.concatenate([ja1, ja2], axis=0)
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7],
                     [0, 1, 2, 3],
                     [4, 5, 6],
                     [7, 8]])

        >>> _.shape
        (6, (3, 2, 3, 4, 3, 2))

        >>> jagged.concatenate([ja, ja], axis=1)
        JaggedArray([[0, 1, 2, 0, 1, 2, 3],
                     [3, 4, 4, 5, 6],
                     [5, 6, 7, 7, 8]])

        >>> _.shape
        (3, (6, 4, 6))

    """
    axis = sanitize_axis(axis, jarrs[0].ndim, multi=False)
    if not len({arr.ndim for arr in jarrs}) == 1:
        raise ValueError("all the input arrays must have same number of dimensions")

    if axis == 0:
        return JaggedArray(
            np.concatenate([jarr.ravel() for jarr in jarrs]),
            shapes=np.concatenate([jarr.shapes for jarr in jarrs]),
        )
    else:
        if not all(
            len(set(shapes)) == 1
            for i, shapes in enumerate(zip(*(jarr.shape for jarr in jarrs)))
            if i != axis
        ):
            msg = "all the input array dimensions except for the concatenation axis must match exactly"
            raise ValueError(msg)
        return JaggedArray.from_iliffe(
            [np.concatenate(arrs, axis=axis - 1) for arrs in zip(*jarrs)]
        )


def stack(jarrs: Iterable[JaggedArray], axis: int = -1) -> JaggedArray:
    """ Stack JaggedArrays on a new axis.

    Args:
        jarrs:
            The jagged arrays to stack.

        axis:
            The axis in the result array along which the arrays are stacked.

    Notes:
        It is not possible to stack along the 0'th axis, as this is the jagged
        inducing dimension.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray
        >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))

        >>> jagged.stack([ja, ja])
        JaggedArray([[[0, 0],
                      [1, 1],
                      [2, 2]],
        <BLANKLINE>
                     [[3, 3],
                      [4, 4]],
        <BLANKLINE>
                     [[5, 5],
                      [6, 6],
                      [7, 7]]])

        >>> _.shape
        (3, (3, 2, 3), 2)

        >>> jagged.stack([ja, ja], axis=1)
        JaggedArray([[[0, 1, 2],
                      [0, 1, 2]],
        <BLANKLINE>
                     [[3, 4],
                      [3, 4]],
        <BLANKLINE>
                     [[5, 6, 7],
                      [5, 6, 7]]])

        >>> _.shape
        (3, 2, (3, 2, 3))

        >>> jagged.stack([ja, ja], axis=0)
        Traceback (most recent call last):
            ...
        ValueError: cannot stack over the jagged inducing dimension
    """
    axis = sanitize_axis(axis, jarrs[0].ndim + 1)
    if axis == 0:
        raise ValueError("cannot stack over the jagged inducing dimension")
    else:
        return concatenate([expand_dims(jarr, axis=axis) for jarr in jarrs], axis=axis)


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
    jarr: JaggedArray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: Optional[DtypeLike] = None,
    out: Optional[np.ndarray] = None,
):
    """ Return the sum along diagonals of a jagged array.

    Args:
        jarr:
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


def resize(jarr: JaggedArray, shape: JaggedShapeLike):
    """ Resize a jagged array.

    Args:
        jarr:
            The jagged array to resize.

        shape:
            The shape of the resized array

    Notes:
        This will fill the new array with repeated copies of the array.
        Not this is different from :meth:`JaggedArray.resize`

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray
        >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))

        >>> jagged.resize(ja, (2, (3, 2)))
        JaggedArray([[0, 1, 2],
                     [3, 4]])

        >>> jagged.resize(ja, (3, (3, 4, 3)))
        JaggedArray([[0, 1, 2],
                     [3, 4, 5, 6],
                     [7, 0, 1]])

        >>> ja = JaggedArray(np.arange(3), (2, (1, 2)))
        >>> jagged.resize(ja, (3, (3, 2, 3)))
        JaggedArray([[0, 1, 2],
                     [0, 1],
                     [2, 0, 1]])
    """
    return JaggedArray(shape, np.resize(jarr.data, shape.size))


def flatten(jarr):
    """ Flatten the jagged array.

    This creates a **copy** of the data.

    Args:
        The jagged array to flatten.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray

        >>> jarr = JaggedArray(np.arange(8), (3, (3, 2, 3)))
        >>> flattened = jagged.flatten(jarr)
        >>> flattened
        array([0, 1, 2, 3, 4, 5, 6, 7])

        >>> flattened[...] = 0
        >>> jarr
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

    See Also:
        JaggedArray.ravel
        JaggedArray.flatten
        jagged.ravel
    """
    return np.asarray(ascontiguousarray(jarr, copy=True).data)


def ravel(jarr):
    """ Ravel the array.

    Creates a **view** of the data.

    Args:
        jarr:
            the jagged array to ravel

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray

        >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
        >>> ravelled = jagged.ravel(ja)
        >>> ravelled
        array([0, 1, 2, 3, 4, 5, 6, 7])
        >>> ravelled[...] = 0
        >>> ja
        JaggedArray([[0, 0, 0],
                     [0, 0],
                     [0, 0, 0]])

    See Also:
        JaggedArray.ravel
        jagged.flatten
        jagged.ravel
    """
    return np.asarray(ascontiguousarray(jarr, copy=False).data)


def digitize(jarr, bins: ArrayLike, right: bool = False) -> JaggedArray:
    """ Return the indices of the bins for each value in array.

    Args:
        bins:
            Array of 1-dimensional, monotonic bins.

        right:
            Whether the intervals include the right or the left bin edge.

    Examples:
        >>> import numpy as np
        >>> import jagged
        >>> from jagged import JaggedArray

        >>> jagged.digitize(
        ...     JaggedArray(np.arange(8), shape=(3, (3, 2, 3))),
        ...     [2, 4, 7]
        ... )
        JaggedArray([[0, 0, 1],
                        [1, 2],
                        [2, 2, 3]])
    """
    return JaggedArray(np.digitize(jarr.data, bins, right=right), jarr.shape)


def smoothe(jarr, axis: AxisLike = None):
    """ smoothe a jagged axis by removing jagged ends.

    Args:
        jarr:
            the jagged axis.
        axis:
            the axis to smoothe.  When passed `None`, smoothe all axes and
            return a numpy array.

    Examples:
        >>> jagged.arange(shape=(3, (3, 2, 3))).smoothe()
        array([[0, 1],
               [3, 4],
               [5, 6]])

        >>> jagged.arange(shape=(3, (3, 2, 3))).smoothe(axis=1)
        JaggedArray([[0, 1],
                     [3, 4],
                     [5, 6]])

        >>> jagged.arange(shape=(3, (3, 2, 3), (2, 3, 2))).smoothe(axis=(1, 2))
        JaggedArray([[[ 0,  1],
                      [ 2,  3]],

                      [[ 6,  7],
                       [ 9, 10]],

                      [[12, 13],
                       [14, 15]]])

        >>> jagged.arange(shape=(3, (3, 2, 3))).smoothe(axis=0)
        Traceback (most recent call last):
            ...
        ValueError: axis 0 is not jagged and so cannot be smoothed.

    """
    if axis is None:
        return jarr[tuple(slice(None, ax) for ax in jarr.minshape)].to_array()

    if is_integer(axis):
        axis = (axis,)

    for ax in axis:
        if ax not in jarr.jagged_axes:
            raise ValueError(f"axis {ax} is not jagged and so cannot be smoothed.")
    index = tuple(
        slice(None, ax if i in jarr.jagged_axes else None)
        for i, ax in enumerate(jarr.minshape)
    )
    return jarr[index]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
