#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.masked
~~~~~~~~~~~~~

Support for converting jagged arrays to and from numpy masked arrays.
"""
import numpy as np

from jagged.core import JaggedArray


def _mask(arr) -> np.ndarray:
    """ the mask for a dense array for the given shapes. """
    mask = np.ones(arr.limits, dtype=bool)
    for ax, shape, limit in zip(range(1, len(arr.limits)), arr.shape, arr.limits[1:]):
        ax_mask = np.arange(limit) < np.expand_dims(shape, 1)
        new_shape = np.ones(len(arr.limits), dtype=int)
        new_shape[0], new_shape[ax] = arr.limits[0], limit
        mask = mask & ax_mask.reshape(*new_shape)
    return mask


def from_masked(arr: np.ma.masked_array) -> JaggedArray:
    raise NotImplementedError


def to_masked(arr: JaggedArray) -> np.ma.masked_array:
    raise NotImplementedError
