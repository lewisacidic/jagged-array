#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged shapes. """
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest
from pytest import raises
from pytest import warns

from ..shape import JaggedShape


@pytest.mark.parametrize(
    ("shape", "desired", "expectation"),
    [
        ((2, (1, 2)), (2, (1, 2)), does_not_raise()),
        ([2, [1, 2]], (2, (1, 2)), does_not_raise()),
        (np.array([2, [1, 2]]), (2, (1, 2)), does_not_raise()),
        ((5, (1, 2, 3, 4, 5)), (5, (1, 2, 3, 4, 5)), does_not_raise()),
        ((2, (2, 2), (1, 2)), (2, 2, (1, 2)), does_not_raise()),
        (JaggedShape((2, (1, 2))), (2, (1, 2)), does_not_raise()),
        ((3, (1, 2)), (2, (1, 2)), raises(ValueError)),
        ((3.5, (1, 2)), None, raises(ValueError)),
        ((3, (1.5, 2)), None, raises(ValueError)),
        ((3, (1, 2), 1.5), None, raises(ValueError)),
        ((2, 2), (2, 2), warns(Warning)),
        ((2, (2, 2)), (2, 2), warns(Warning)),
        ((1, (2,)), (1, 2), warns(Warning)),
    ],
    ids=[
        "shape as tuples",
        "shape as lists",
        "shape as array",
        "longer shape",
        "collapse jagged appearing",
        "shape as shape",
        "bad shape",
        "inducing length is float",
        "jagged length is float",
        "flat length is float",
        "non-jagged shape",
        "jagged-appearing but not",
        "inducing length is one",
    ],
)
def test_init(shape, desired, expectation):
    with expectation:
        assert JaggedShape(shape) == desired


@pytest.mark.parametrize(
    ["shape", "desired"],
    [
        ((2, (1, 2)), [[1], [2]]),
        ((3, 2, (1, 2, 3)), [[2, 1], [2, 2], [2, 3]]),
        ((3, (1, 2, 3), (3, 2, 1)), [[1, 3], [2, 2], [3, 1]]),
        ((3, (3, 2, 3), (1, 2, 3), (3, 1, 2)), [[3, 1, 3], [2, 2, 1], [3, 3, 2]]),
        (
            (5, (4, 2, 1, 5, 3), (2, 4, 6, 1, 2)),
            [[4, 2], [2, 4], [1, 6], [5, 1], [3, 2]],
        ),
    ],
)
def test_to_shapes(shape, desired):
    np.testing.assert_equal(JaggedShape(shape).to_shapes(), desired)


@pytest.mark.parametrize(
    ["shape", "desired"],
    [
        ((2, (1, 2)), 3),
        ((3, 2, (1, 2, 3)), 12),
        ((3, (1, 2, 3), (3, 2, 1)), 10),
        ((5, (4, 2, 1, 5, 3), (2, 4, 6, 1, 2)), 33),
    ],
)
def test_size(shape, desired):
    assert JaggedShape(shape).size == desired


@pytest.mark.parametrize(
    ["shape", "desired"],
    [
        ((2, (1, 2)), (1, 2)),
        ((3, 2, (1, 2, 3)), (2, 4, 6)),
        ((3, (1, 2, 3), (3, 2, 1)), (3, 4, 3)),
        ((5, (4, 2, 1, 5, 3), (2, 4, 6, 1, 2)), (8, 8, 6, 5, 6)),
    ],
)
def test_sizes(shape, desired):
    assert JaggedShape(shape).sizes == desired


@pytest.mark.parametrize(
    ["shape", "desired"],
    [
        ((2, (1, 2)), 2),
        ((3, 2, (1, 2, 3)), 3),
        ((3, (1, 2, 3), (3, 2, 1)), 3),
        ((5, (4, 2, 1, 5, 3), (2, 4, 6, 1, 2)), 3),
    ],
)
def test_ndim(shape, desired):
    assert JaggedShape(shape).ndim == desired


@pytest.mark.parametrize(
    ["shape", "desired"],
    [
        ((3, (1, 2, 3)), (3, 3)),
        ((3, 2, (1, 2, 3)), (3, 2, 3)),
        ((3, (2, 2, 2), (1, 2, 3)), (3, 2, 3)),
    ],
)
def test_limits(shape, desired):
    assert JaggedShape(shape).limits == desired


@pytest.mark.parametrize(
    ["shape", "desired"],
    [
        ((2, (1, 2)), (1,)),
        ((3, 2, (1, 2, 3)), (2,)),
        ((3, (1, 2, 3), (3, 2, 1)), (1, 2)),
        ((3, (1, 2, 3), 2, (3, 2, 3)), (1, 3)),
        ((3, (2, 2, 2), (1, 4, 3)), (2,)),
    ],
)
def test_jagged_axes(shape, desired):
    assert JaggedShape(shape).jagged_axes == desired


@pytest.mark.parametrize(
    ("shapes", "desired", "expectation"),
    [
        ([[1], [2]], (2, (1, 2)), does_not_raise()),
        (((1,), (2,)), (2, (1, 2)), does_not_raise()),
        (np.array([[1], [2]]), (2, (1, 2)), does_not_raise()),
        ([[5, 3], [2, 6], [2, 6]], (3, (5, 2, 2), (3, 6, 6)), does_not_raise()),
        ([[2], [2]], (2, 2), warns(Warning)),
        ([[1.5], [2.5]], None, raises(ValueError)),
    ],
    ids=[
        "shapes as tuples",
        "shapes as lists",
        "shapes as arrays",
        "longer shapes",
        "non jagged shape",
        "shapes are floats",
    ],
)
def test_from_shapes(shapes, desired, expectation):
    with expectation:
        assert JaggedShape.from_shapes(shapes) == desired
