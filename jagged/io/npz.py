#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
import numpy as np

from ..jagged import JaggedArray
from ..typing import FileLike


def save_npz(filename: FileLike, array: JaggedArray, compressed=True) -> None:
    """ Save a jagged array to disk using numpy's `.npz` format.

    Args:
        filename: the file to which to write
        array: the jagged array to save
        compressed: whether to compress the array

    Examples:
        >>> import jagged
        >>> arr = jagged.JaggedArray(np.arange(22), [[3, 2, 3], [3, 2, 3]])
        >>> arr
        JaggedArray(data=[ 0  1  2 ... 19 20 21],
            shape=[[3 2 3]
                   [3 2 3]],
            dtype=int64)
        >>> jagged.save_npz('jagged.npz', arr)
        >>> loaded = jagged.load_npz('jagged.npz')
        >>> loaded
        JaggedArray(data=[ 0  1  2 ... 19 20 21],
            shape=[[3 2 3]
                   [3 2 3]],
            dtype=int64)

    See Also:
        load_npz
    """

    nodes = {"data": array.data, "shape": array.shape}
    if compressed:
        np.savez_compressed(filename, **nodes)
    else:
        np.savez(filename, **nodes)


def load_npz(filename: FileLike) -> JaggedArray:
    """ Load a jagged array in numpy's `npz` format from disk.

    Args:
        filename: The file to read.

    See Also:
        save_npz
    """

    with np.load(filename) as f:
        try:
            data = f["data"]
            shape = f["shape"]
            return JaggedArray(data, shape)
        except KeyError:
            msg = "The file {!r} does not contain a valid jagged array".format(filename)
            raise RuntimeError(msg)
