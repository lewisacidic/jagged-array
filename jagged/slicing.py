#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.slicing
~~~~~~~~~~~~~~

Slicing support for jagged-array.
"""
from typing import Tuple

import numpy as np

from .shape import JaggedShape
from .typing import Index
from .typing import IndexLike
from .typing import JaggedShapeLike
from .utils import is_integer
from .utils import is_iterable


def canonicalize_index(index: IndexLike, shape: JaggedShapeLike):
    """ Convert an index into its canonical form for a given shape array.

    Procedure:
        1) convert index to a tuple
        2) expand the index to cover all dimensions
    """

    shape = JaggedShape(shape)
    index = sanitize_index(index)
    index = expand_index(index, shape.ndim)
    index = sum(
        (
            canonicalize_subindex(*aligned)
            for aligned in align_index_shapes(index, shape)
        ),
        (),
    )

    return index


def align_index_shapes(index: Index, shape: JaggedShapeLike):
    """ align a complete index against a slicing shape.

    This will produce a generator of pairs of index with tuples of their aligned shapes.
    """
    ax = 0
    for ix in index:
        if hasattr(ix, "ndim") and ix.dtype == bool:
            yield ix, shape[ax : ax + ix.ndim], ax
            ax += ix.ndim
        elif ix is None:
            yield ix, 0, ax
        else:
            yield ix, shape[ax : ax + 1], ax
            ax += 1


def sanitize_index(index: IndexLike) -> Index:
    """ sanitize an index.

    This will convert it to a tuple, and turn all iterables to numpy arrays.

    Args:
        index:
            The index to sanitize.
    Examples:
        >>> sanitize_index(1)
        (1,)

        >>> sanitize_index([2, 3])
        (2, 3)

        >>> sanitize_index((2, [4, 2]))
        (2, array([4, 2]))

        >>> sanitize_index([None, slice(2, 3, 1)])
        (None, slice(2, 3, 1))

        >>> sanitize_index([[2, 4], 6, 8]])
        array([[2, 4],
               [6, 8]])
    """

    if not isinstance(index, tuple):
        index = (index,)
    return tuple(np.array(ix) if is_iterable(ix) else ix for ix in index)


def expand_index(index: Tuple, n_dims: int) -> Tuple:
    """ Expand the index with empty slices.

    This will expand ellipses if present, or pad at the end if not.

    Args:
        index:
            The index to expand
        n_dims:
            The number of dimensions of the indexed object

    This fills in missing dimensions shown with ellipsis with full slices.

    Examples:
        >>> expand_index((5, 3), 4)
        (5, 3, slice(None, None, None), slice(None, None, None))

        >>> expand_index((5, Ellipsis, 3), 4)
        (5, slice(None, None, None), slice(None, None, None), 3)

        >>> expand_index((5, 3), 2)
        (5, 3)

        >>> expand_index((Ellipsis,), 2)
        (slice(None, None, None), slice(None, None, None))

        >>> expand_index((1, None, 2), 2)
        (1, None, 2)

        >>> expand_index((Ellipsis, None), 2)
        (slice(None, None, None), slice(None, None, None), None)

        >>> expand_index((np.array([1, 2, 3]),), 2)
        (array([1, 2, 3]), slice(None, None, None))

        >>> expand_index((np.array([[1, 2, 3], [4, 5, 6]]),), 2)
        (array([[1, 2, 3],
                [4, 5, 6]]), slice(None, None, None))
    """

    ellipses = [i for i, ix in enumerate(index) if ix is Ellipsis]

    if len(ellipses) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    elif len(ellipses) == 0:
        eix = len(index)
        index = (*index, Ellipsis)
    else:
        eix = ellipses[0]

    dims_to_add = n_dims + 1
    for ix in index:
        if isinstance(ix, np.ndarray) and ix.dtype == bool:
            dims_to_add -= ix.ndim
        elif ix is not None:
            dims_to_add -= 1

    return (*index[:eix], *(slice(None, None, None),) * dims_to_add, *index[1 + eix :])


def canonicalize_subindex(index_dim, shape_dims, ax=0):
    """ canonicalize a subindex for its matching shapes """

    if index_dim is None:
        return (None,)

    elif is_integer(index_dim):
        shape_dim = shape_dims[0]
        lim = shape_dim if is_integer(shape_dim) else max(shape_dim)

        if index_dim > lim or index_dim < -lim:
            raise IndexError(
                f"index {index_dim} is out of bounds for axis {ax} with size {lim}"
            )
        return (posify(index_dim, lim),)

    elif isinstance(index_dim, slice):
        shape_dim = shape_dims[0]
        lim = shape_dim if is_integer(shape_dim) else max(shape_dim)

        start, stop, step = index_dim.start, index_dim.stop, index_dim.step
        step = 1 if step is None else step

        if step > 0:
            start = 0 if start is None else inrange(posify(start, lim), 0, lim)
            stop = lim if stop is None else inrange(posify(stop, lim), 0, lim)
            if start > stop:
                start = stop

        else:
            start = (
                lim - 1 if start is None else inrange(posify(start, lim), -1, lim - 1)
            )
            stop = -1 if stop is None else inrange(posify(stop, lim), -1, lim - 1)
            if start < stop:
                start = stop
        return (slice(start, stop, step),)

    else:
        # is a numpy array
        if index_dim.dtype == bool:
            # shape is iterable of dimensions absorbed by boolean mask
            for i, (bool_dim, shape_dim) in enumerate(zip(index_dim.shape, shape_dims)):
                if bool_dim != shape_dim:
                    msg = f"boolean index did not match indexed array along dimension {ax + i}; dimension is {shape_dim} but corresponding boolean dimension is {bool_dim}"
                    raise IndexError(msg)
            return index_dim.nonzero()
        else:
            shape_dim = shape_dims[0]
            lim = shape_dim if is_integer(shape_dim) else max(shape_dim)
            dtype = index_dim.dtype
            if not np.issubdtype(dtype, np.signedinteger):
                msg = f"arrays used as indices must be of integer (or boolean) type. Was {dtype}"
                raise IndexError(msg)

            min_, max_ = np.min(index_dim), np.max(index_dim)
            oob = max_ if max_ > lim else min_ if min_ < -lim else None
            if oob:
                msg = f"index {oob} is out of bounds for axis {ax} with size {lim}"
                raise IndexError(msg)
            else:
                return (np.where(index_dim > 0, index_dim, index_dim + lim),)


def posify(ix, lim):
    return ix if ix > 0 else ix + lim


def inrange(x, lo, hi):
    return max(lo, min(x, hi))
