#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged
~~~~~~

Jagged array support for the pydata ecosystem.
"""
from .api import arange
from .api import array_equal
from .api import concatenate
from .api import diagonal
from .api import digitize
from .api import empty
from .api import empty_like
from .api import expand_dims
from .api import flatten
from .api import full
from .api import full_like
from .api import ones
from .api import ones_like
from .api import random
from .api import ravel
from .api import reshape
from .api import resize
from .api import smoothe
from .api import squeeze
from .api import stack
from .api import where
from .api import zeros
from .api import zeros_like
from .core import JaggedArray
from .iliffe import iliffe_to_jagged as from_iliffe
from .masked import masked_to_jagged as from_masked

__all__ = [
    "JaggedArray",
    "arange",
    "concatenate",
    "diagonal",
    "expand_dims",
    "flatten",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "empty",
    "empty_like",
    "digitize",
    "array_equal",
    "random",
    "ravel",
    "reshape",
    "resize",
    "squeeze",
    "stack",
    "where",
    "zeros",
    "zeros_like",
    "JaggedArray",
    "from_iliffe",
    "from_masked",
    "smoothe",
]
