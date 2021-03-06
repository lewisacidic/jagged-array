#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.io
~~~~~~~~~

IO functionality for jagged arrays.
"""
from .npz import load_npz
from .npz import save_npz

__all__ = ["load_npz", "save_npz"]
