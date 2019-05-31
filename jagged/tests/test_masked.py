#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged to/from masked operations. """
import numpy as np
import pytest

from ..api import arange
from ..api import random
from ..core import JaggedArray
from ..masked import jagged_to_masked
from ..masked import masked_to_jagged
from .utils import assert_equal
from .utils import assert_masked_equal


@pytest.mark.parametrize(
    "func",
    [
        masked_to_jagged,
        JaggedArray.from_masked,
        pytest.param(JaggedArray, marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize(
    ["mskd", "desired"],
    [
        (
            np.ma.masked_array(
                [[0, 1, 0], [2, 3, 4], [5, 6, 0]],
                [[False, False, True], [False, False, False], [False, False, True]],
            ),
            arange(shape=(3, (2, 3, 2))),
        )
    ],
)
def test_from_masked(mskd, desired, func):
    assert_equal(func(mskd), desired)


@pytest.mark.parametrize("func", [jagged_to_masked, lambda x: x.to_masked()])
@pytest.mark.parametrize(
    ["jarr", "desired"],
    [
        (
            arange(shape=(3, (2, 3, 2))),
            np.ma.masked_array(
                [[0, 1, 0], [2, 3, 4], [5, 6, 0]],
                [[False, False, True], [False, False, False], [False, False, True]],
            ),
        )
    ],
)
def test_to_masked(func, jarr, desired):
    assert_masked_equal(func(jarr), desired)


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
