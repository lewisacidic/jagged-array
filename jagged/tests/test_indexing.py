#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for indexing operations on jagged arrays. """
import numpy as np
import pytest
from pytest import param

from ..api import random
from ..core import JaggedArray
from .utils import assert_equal
from .utils import does_not_raise
from .utils import raises


INDEX_OPS = [
    param(0, does_not_raise(), id="first"),
    param(1, does_not_raise(), id="middle"),
    param(-1, does_not_raise(), id="last"),
    param(-2, does_not_raise(), id="last but one"),
    param(7, raises(IndexError), id="misses forward"),
    param(-7, raises(IndexError), id="misses backward"),
    param(slice(0, 0), does_not_raise(), id="slice width zero"),
    param(slice(0, 1), does_not_raise(), id="slice width one"),
    param(slice(0, 2), does_not_raise(), id="slice width two"),
    param(slice(0, -1), does_not_raise(), id="slice negative stop"),
    param(slice(0, -2), does_not_raise(), id="slice negative stop above 1"),
    param(slice(-3, 4), does_not_raise(), id="slice negative start"),
    param(slice(-3, -1), does_not_raise(), id="slice negative start and stop"),
    param(slice(0, 5, 2), does_not_raise(), id="slice step > 1"),
    param(slice(1, 0, -1), does_not_raise(), id="slice negative step"),
    param(slice(-1, -2, -1), does_not_raise(), id="slice all negative"),
    param(slice(0, 10), does_not_raise(), id="oversized slice"),
    param([0, 1], does_not_raise(), id="multi index"),
    param([0, -1], does_not_raise(), id="multi index with negative"),
    param([False, True, True, False, True], does_not_raise(), id="boolean index"),
    param([False, True], raises(IndexError), id="undersized boolean index"),
    param([False, True] * 5, raises(IndexError), id="oversized boolean index"),
    param(np.array([[2, 1], [3, 1]]), does_not_raise(), id="2D index"),
]


@pytest.mark.parametrize(["index", "expectation"], INDEX_OPS)
def test_inducing_index(index, expectation):
    jarr = random((5, 5, 5))
    with expectation:
        assert_equal(jarr[index], JaggedArray.from_masked(jarr.to_masked()[index]))
