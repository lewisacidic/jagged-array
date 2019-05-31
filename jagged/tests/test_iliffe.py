#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged to/from iliffe operations. """
import numpy as np
import pytest

from ..api import arange
from ..api import random
from ..core import JaggedArray
from ..iliffe import iliffe_to_jagged
from ..iliffe import jagged_to_iliffe
from .utils import assert_equal
from .utils import assert_iliffe_equal


@pytest.mark.parametrize(
    "func",
    [
        iliffe_to_jagged,
        JaggedArray.from_iliffe,
        pytest.param(JaggedArray, marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize(
    ["ivec", "desired"],
    [
        ([[0, 1], [2, 3, 4], [5, 6]], arange(shape=(3, (2, 3, 2)))),
        (((0, 1), (2, 3, 4), (5, 6)), arange(shape=(3, (2, 3, 2)))),
        (
            np.array([np.array([0, 1]), np.array([2, 3, 4]), np.array([5, 6])]),
            arange(shape=(3, (2, 3, 2))),
        ),
        # causes numpy to try to broadcast, see np.array([np.array([[0, 1]]), np.array([[2]])])
        ([[[0, 1]], [[2]]], arange(shape=(2, 1, (2, 1)))),
        ([[0, 1.0], [2, 3, 4], [5, 6]], arange(shape=(3, (2, 3, 2)), dtype=float)),
    ],
    ids=["with lists", "with tuples", "with ndarray", "expdim 1", "detect dtype"],
)
def test_from_iliffe(ivec, desired, func):
    result = func(ivec)
    assert_equal(result, desired)
    assert result.dtype == desired.dtype


ax1dim1 = np.empty(2, dtype=object)
ax1dim1[...] = [[[0, 1]], [[2]]]


@pytest.mark.parametrize("func", [jagged_to_iliffe, lambda x: x.to_iliffe()])
@pytest.mark.parametrize(
    ["jarr", "desired"],
    [
        (
            arange(shape=(3, (2, 3, 2))),
            np.array([np.array([0, 1]), np.array([2, 3, 4]), np.array([5, 6])]),
        ),
        (
            arange(shape=(3, (4, 2, 3), (3, 4, 3))),
            np.array(
                [
                    np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),
                    np.array([[12, 13, 14, 15], [16, 17, 18, 19]]),
                    np.array([[20, 21, 22], [23, 24, 25], [26, 27, 28]]),
                ]
            ),
        ),
        (arange(shape=(2, 1, (2, 1))), ax1dim1),
    ],
)
def test_to_iliffe(func, jarr, desired):
    assert_iliffe_equal(func(jarr), desired)


@pytest.mark.parametrize("dtype", ["f8", "f4", "i8", "i4"])
@pytest.mark.parametrize(
    "shape",
    [
        (3, (3, 2, 3)),
        (3, (3, 2, 3), 2),
        (3, (3, 2, 3), (3, 2, 3)),
        (3, (3, 2, 3), 2, (2, 3, 2)),
        (3, (3, 2, 3), 2, (2, 3, 2), 2),
        (3, 1, (3, 2, 3)),
        (3, (3, 2, 3), 1),
        (3, 1, 1, (3, 2, 3)),
    ],
    ids=[
        "one jagged",
        "one jagged one flat",
        "two jagged",
        "two jagged one flat",
        "two jagged two flat",
        "expdim 1",
        "expdim 2",
        "expdim 1,2",
    ],
)
def test_round_trip(shape, dtype):
    jarr = random(shape, dtype=dtype, random_state=42)
    result = JaggedArray.from_iliffe(jarr.to_iliffe())
    assert_equal(jarr, result)
