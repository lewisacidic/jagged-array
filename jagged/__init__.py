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
from .api import concatenate
from .api import diagonal
from .api import expand_dims
from .api import full
from .api import full_like
from .api import ones
from .api import ones_like
from .api import random
from .api import squeeze
from .api import stack
from .api import trace
from .api import where
from .api import zeros
from .api import zeros_like
from .core import JaggedArray

__all__ = [
    "JaggedArray",
    "random",
    "concatenate",
    "squeeze",
    "expand_dims",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "stack",
    "trace",
    "diagonal",
    "where",
]
