#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for indexing of jagged arrays. """
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest
from pytest import raises
from pytest import warns

from ..indexing import JaggedShape


@pytest.mark.parametrize(
    ("shape", "expectation"),
    [
        ((2, (1, 2)), does_not_raise()),
        ([2, [1, 2]], does_not_raise()),
        (np.array([2, [1, 2]]), does_not_raise()),
        ((3, (1, 2)), raises(ValueError)),
        ((3.5, (1, 2)), raises(ValueError)),
        ((3, (1.5, 2)), raises(ValueError)),
        ((3, (1, 2), 1.5), raises(ValueError)),
        ((3, 3), warns(Warning)),
    ],
    ids=[
        "shape as tuples",
        "shape as lists",
        "shape as array",
        "bad shape",
        "inducing length is float",
        "jagged length is float",
        "flat length is float",
        "non-jagged shape",
    ],
)
def test_init(shape, expectation):
    with expectation:
        assert JaggedShape(shape).shape == (2, (1, 2))


@pytest.mark.parametrize(
    ("shapes", "expectation"),
    [
        ([[1], [2]], does_not_raise()),
        (((1,), (2,)), does_not_raise()),
        (np.array([[1], [2]]), does_not_raise()),
        ([[2], [2]], warns(Warning)),
        ([[1.5], [2.5]], raises(ValueError)),
    ],
    ids=[
        "shapes as tuples",
        "shapes as lists",
        "shapes as arrays",
        "non jagged shape",
        "shapes are floats",
    ],
)
def test_with_shapes(shapes, expectation):
    with expectation:
        assert JaggedShape.from_shapes(shapes).shape == (2, (1, 2))
