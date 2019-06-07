#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.indexing
~~~~~~~~~~~~~~~

Support for indexing jagged arrays.
"""
from math import ceil

import numpy as np

from .utils import array_to_metadata
from .utils import shapes_to_shape


def getitem(arr, index):
    """ index a given array with a given index """

    ind_ix, *ixs = index
    squeeze_ind = False

    if isinstance(ind_ix, int):
        ind_ix = slice(ind_ix, ind_ix + 1, 1)
        squeeze_ind = True

    strides = arr.strides_array[ind_ix]
    offsets = arr.offsets_array[ind_ix]
    shapes = arr.shape_array[ind_ix]

    i = 0
    for ix in ixs:
        if ix is None:
            shapes = np.insert(shapes, i, 1, axis=1)
            strides = np.insert(shapes, i, 0, axis=1)
            i += 1
        elif isinstance(ix, int):
            offsets += ix * strides[:, i]
            shapes = np.delete(shapes, i, axis=1)
            strides = np.delete(strides, i, axis=1)
            i -= 1
        elif isinstance(ix, slice):
            offsets += ix.start * strides[:, i]
            shapes[:, i] = np.clip(
                shapes[:, i], None, ceil(ix.stop - ix.start) / ix.step
            )
            strides[:, i] *= ix.step

        # multi-index support here
        i += 1

    shape = shapes_to_shape(shapes)

    if squeeze_ind:
        return np.ndarray(
            shape=shapes[0],
            buffer=arr.data,
            dtype=arr.dtype,
            offset=offsets[0],
            strides=strides[0],
            order=arr.order,
        )

    else:
        strides = array_to_metadata(strides)

        return arr.__class__(
            shape,
            buffer=arr.data,
            dtype=arr.dtype,
            strides=strides,
            offsets=offsets,
            order=arr.order,
        )
