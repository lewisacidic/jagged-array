#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

from typing import TextIO, BinaryIO, Union
from pathlib import Path

import numpy as np

FileLike = Union[BinaryIO, TextIO, str, Path]
RandomState = Union[np.random.RandomState, int]
Dtype = Union[str, np.dtype]
