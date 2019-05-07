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
from .core import JaggedArray
from .utils import random

__all__ = ["JaggedArray", "random"]
