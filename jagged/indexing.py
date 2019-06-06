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

from .slicing import canonicalize_index


def getitem(arr, item):
    item = canonicalize_index(item, arr.shape)

    ind_ix, *ixs = item
    squeeze_ind = False

    if isinstance(ind_ix, int):
        ind_ix = slice(ind_ix, ind_ix + 1, 1)
        squeeze_ind = True

    strides = arr.strides.copy()[ind_ix]
    offset = arr.offsets.copy()[ind_ix.start]
    shapes = arr.shapes.copy()[ind_ix]

    i = 0
    for ix in ixs:
        if ix is None:
            shapes = np.insert(shapes, i, 1, axis=1)
            strides = np.insert(shapes, i, 0, axis=1)
            i += 1
        elif isinstance(ix, int):
            offset += ix * strides[0, i]
            shapes = np.delete(shapes, i, axis=1)
            strides = np.delete(strides, i, axis=1)
            i -= 1
        elif isinstance(ix, slice):
            offset += ix.start * strides[0, i]
            shapes[:, i] = np.clip(
                shapes[:, i], None, ceil(ix.stop - ix.start) / ix.step
            )
            strides[:, i] *= ix.step

        # multi-index support here
        i += 1

    start = offset // arr.dtype.itemsize
    shape = arr.shape.__class__.from_shapes(shapes)

    data = arr.data[start : start + shape.size]
    print(offset, data, strides, shapes)

    if squeeze_ind:
        return np.ndarray(
            shape=shapes[0],
            offset=offset,
            buffer=arr.data,
            dtype=arr.dtype,
            strides=strides[0],
        )

    else:
        return arr.__class__(data=data, dtype=arr.dtype, strides=strides, shape=shape)
