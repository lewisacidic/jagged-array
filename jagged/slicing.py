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

from .typing import IndexLike
from .typing import ShapeLike


def normalize_index(index: IndexLike, shape: ShapeLike):
    """ Convert an index into its canonical form for a given shape array.

    Procedure:
        1) convert index to tuple
        2) complete the index to cover all dimensions
        3) replace `None` with full slices
        3) check the bounding conditions
    """

    pass


def complete_index(index: Tuple, n_dims: int) -> Tuple:
    """ Complete the index with empty slices.

    This will expand ellipses if present, or pad at the end if not.

    Args:
        index:
            The index in which to replace ellipses
        n_dims:
            The number of dimensions of the sliced object

    This fills in missing dimensions shown with ellipsis with full slices.

    Examples:
        >>> complete_index((5, 3), 4)
        (5, 3, slice(None, None, None), slice(None, None, None))

        >>> complete_index((5, Ellipsis, 3), 4)
        (5, slice(None, None, None), slice(None, None, None), 3)

        >>> complete_index((5, 3), 2)
        (5, 3)

        >>> complete_index((Ellipsis,), 2)
        (slice(None, None, None), slice(None, None, None))

        >>> complete_index((1, None, 2), 2)
        (1, None, 2)

        >>> complete_index((Ellipsis, None), 2)
        (slice(None, None, None), slice(None, None, None), None)
    """

    n_ellipses = sum(ix is Ellipsis for ix in index)

    if n_ellipses > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    if n_ellipses == 0:
        if n_dims == len(index):
            return index
        else:
            index = (*index, Ellipsis)

    ixix = index.index(Ellipsis)
    extra_dims = 1 + n_dims + sum(ix is None for ix in index) - len(index)
    fill = (slice(None, None, None),) * extra_dims
    return (*index[:ixix], *fill, *index[1 + ixix :])
