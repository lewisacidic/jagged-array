#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.factories
~~~~~~~~~~~~~~~~

Factory functions for jagged arrays.
"""
import warnings
from typing import Any
from typing import Optional

import numpy as np

from .core import JaggedArray
from .iliffe import iliffe_to_jagged
from .typing import DtypeLike


def array(
    obj: Any,
    dtype: Optional[DtypeLike] = None,
    copy: Optional[bool] = True,
    order: Optional[str] = "K",
    ndmin: Optional[int] = 2,
    subok: Optional[bool] = False,
):
    if isinstance(obj, JaggedArray):
        pass
    else:
        return iliffe_to_jagged(obj, dtype=dtype)


def ascontiguousarray(jarr, copy=True):
    itemsize = jarr.dtype.itemsize
    if not copy:

        bytesizes = (size * itemsize for size in jarr.sizes[:-1])

        if all(bsize > osize for bsize, osize in zip(bytesizes, np.diff(jarr.offsets))):
            return jarr
        else:
            msg = "Contiguous array impossible without copying: a copy was made despite copy=False."
            warnings.warn(msg)

    buffer = np.concatenate(
        [
            jarr.data[offset // itemsize : offset // itemsize + size]
            for offset, size in zip(jarr.offsets, jarr.sizes)
        ]
    )
    return JaggedArray(
        shape=jarr.shape, buffer=buffer, strides=jarr.strides, dtype=jarr.dtype
    )
