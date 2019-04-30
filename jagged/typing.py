#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

from typing import TextIO, BinaryIO, Union
from collections import Iterable
from pathlib import Path

import numpy as np

FileLike = Union[BinaryIO, TextIO, str, Path]
RandomState = Union[np.random.RandomState, int]
DtypeLike = Union[str, np.dtype]
ArrayLike = Union[np.ndarray, Iterable]
ShapeLike = Union[np.ndarray, Iterable[int]]
AxisLike = Union[int, Iterable[int]]
Number = Union[float, int]
SliceLike = Union[Ellipsis, slice, int, Iterable[Union[Ellipsis, slice, int, bool]]]
