#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Utility functions for writing tests for jagged-array. """
from contextlib import ExitStack as does_not_raise

import numpy as np
from numpy.testing import assert_equal as assert_flat_equal
from pytest import raises
from pytest import warns

from ..api import array_equal


def assert_equal(jarr1, jarr2):
    assert array_equal(jarr1, jarr2)


def assert_iliffe_equal(ivec1, ivec2):
    assert all(np.array_equal(a1, a2) for a1, a2 in zip(ivec1, ivec2))


def assert_masked_equal(marr1, marr2):
    assert np.ma.allequal(marr1, marr2)


__all__ = [
    "does_not_raise",
    "raises",
    "warns",
    "assert_equal",
    "assert_flat_equal",
    "assert_iliffe_equal",
    "assert_masked_equal",
]
