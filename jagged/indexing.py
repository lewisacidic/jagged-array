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
import numpy as np

from .slicing import canonicalize_index


def getitem(arr, item):
    item = canonicalize_index(item)

    zero, *rest = item

    offset = arr.offsets[zero]
    shape = arr.shapes[zero]
    strides = arr.strides[zero]

    return np.ndarray(
        shape=shape, offset=offset, buffer=arr.data, dtype=arr.dtype, strides=strides
    )
