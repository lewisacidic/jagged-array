#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.typing
~~~~~~~~~~~~~

Types for type hints in jagged-array.
"""
from pathlib import Path
from typing import BinaryIO
from typing import Iterable
from typing import TextIO
from typing import Union

import numpy as np

FileLike = Union[BinaryIO, TextIO, str, Path]
RandomState = Union[np.random.RandomState, int]
DtypeLike = Union[str, np.dtype]
ArrayLike = Union[np.ndarray, Iterable]
ShapeLike = Union[np.ndarray, Iterable[int]]
AxisLike = Union[int, Iterable[int]]
Number = Union[float, int]
SliceLike = Union[
    type(Ellipsis), slice, int, Iterable[Union[type(Ellipsis), slice, int, bool]]
]
