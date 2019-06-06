#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for slicing of jagged arrays. """
import numpy as np
import pytest
from pytest import param

from ..slicing import canonicalize_index
from .utils import assert_flat_equal
from .utils import does_not_raise
from .utils import raises


F1D = (10,)
F2D = (10, 10)
J2D = (10, (3, 5, 4, 6, 4, 3, 7, 3, 1, 8))

ZERO_D_OBJS = [
    param(lambda x: x, id="integer"),
    param(lambda x: [x], id="list"),
    param(lambda x: ((x,),), id="tuple"),
    param(lambda x: np.array(x), id="array"),
]


@pytest.mark.parametrize("object", ZERO_D_OBJS)
@pytest.mark.parametrize(
    ["index", "shape", "desired", "expectation"],
    [
        param(5, F1D, (5,), does_not_raise(), id="standard"),
        param(0, F1D, (0,), does_not_raise(), id="low"),
        param(9, F1D, (9,), does_not_raise(), id="high"),
        param(-3, F1D, (7,), does_not_raise(), id="negative"),
        param(-1, F1D, (9,), does_not_raise(), id="negative low"),
        param(-10, F1D, (0,), does_not_raise(), id="negative high"),
        param(1.5, F1D, None, raises(IndexError), id="float"),
        param("c", F1D, None, raises(IndexError), id="string"),
        param(10, F1D, None, raises(IndexError), id="just too high"),
        param(20, F1D, None, raises(IndexError), id="too high"),
        param(-11, F1D, None, raises(IndexError), id="just too low"),
        param(-21, F1D, None, raises(IndexError), id="too low"),
        param(5, F2D, (5, slice(0, 10, 1)), does_not_raise(), id="2d standard"),
        param(0, F2D, (0, slice(0, 10, 1)), does_not_raise(), id="2d low"),
        param(9, F2D, (9, slice(0, 10, 1)), does_not_raise(), id="2d high"),
        param(-3, F2D, (7, slice(0, 10, 1)), does_not_raise(), id="2d negative"),
        param(-1, F2D, (9, slice(0, 10, 1)), does_not_raise(), id="2d negative low"),
        param(-10, F2D, (0, slice(0, 10, 1)), does_not_raise(), id="2d negative high"),
        param(1.5, F2D, None, raises(IndexError), id="2d float"),
        param("c", F2D, None, raises(IndexError), id="2d string"),
        param(10, F2D, None, raises(IndexError), id="2d just too high"),
        param(20, F2D, None, raises(IndexError), id="2d too high"),
        param(-11, F2D, None, raises(IndexError), id="2d just too low"),
        param(-21, F2D, None, raises(IndexError), id="2d too low"),
        param(5, J2D, (5, slice(0, 8, 1)), does_not_raise(), id="jagged standard"),
        param(0, J2D, (0, slice(0, 8, 1)), does_not_raise(), id="jagged low"),
        param(9, J2D, (9, slice(0, 8, 1)), does_not_raise(), id="jagged high"),
        param(-3, J2D, (7, slice(0, 8, 1)), does_not_raise(), id="jagged negative"),
        param(-1, J2D, (9, slice(0, 8, 1)), does_not_raise(), id="jagged negative low"),
        param(
            -10, J2D, (0, slice(0, 8, 1)), does_not_raise(), id="jagged negative high"
        ),
        param(1.5, J2D, None, raises(IndexError), id="float"),
        param("c", J2D, None, raises(IndexError), id="string"),
        param(10, J2D, None, raises(IndexError), id="just too high"),
        param(20, J2D, None, raises(IndexError), id="too high"),
        param(-11, J2D, None, raises(IndexError), id="just too low"),
        param(-21, J2D, None, raises(IndexError), id="too low"),
    ],
)
def test_index_first_axis(object, index, shape, desired, expectation):
    with expectation:
        assert_flat_equal(canonicalize_index(object(index), shape), desired)


@pytest.mark.parametrize(
    ["index", "shape", "desired"],
    [
        param(slice(5, None), F1D, (slice(5, 10, 1),), id="1d start"),
        param(slice(5), F1D, (slice(0, 5, 1),), id="1d stop"),
        param(slice(None, None, 3), F1D, (slice(0, 10, 3),), id="1d step"),
        param(slice(3, 5), F1D, (slice(3, 5, 1),), id="1d start stop"),
        param(slice(3, 5, 2), F1D, (slice(3, 5, 2),), id="1d start stop step"),
        param(slice(5, None), F2D, (slice(5, 10, 1), slice(0, 10, 1)), id="2d start"),
        param(slice(5), F2D, (slice(0, 5, 1), slice(0, 10, 1)), id="2d stop"),
        param(
            slice(None, None, 3), F2D, (slice(0, 10, 3), slice(0, 10, 1)), id="2d step"
        ),
        param(slice(3, 5), F2D, (slice(3, 5, 1), slice(0, 10, 1)), id="2d start stop"),
        param(
            slice(3, 5, 2),
            F2D,
            (slice(3, 5, 2), slice(0, 10, 1)),
            id="2d start stop step",
        ),
        param(
            slice(5, None), J2D, (slice(5, 10, 1), slice(0, 8, 1)), id="jagged start"
        ),
        param(slice(5), J2D, (slice(0, 5, 1), slice(0, 8, 1)), id="jagged stop"),
        param(
            slice(None, None, 3),
            J2D,
            (slice(0, 10, 3), slice(0, 8, 1)),
            id="jagged step",
        ),
        param(
            slice(3, 5), J2D, (slice(3, 5, 1), slice(0, 8, 1)), id="jagged start stop"
        ),
        param(
            slice(3, 5, 2),
            J2D,
            (slice(3, 5, 2), slice(0, 8, 1)),
            id="jagged start stop step",
        ),
    ],
)
def test_slice_first_axis(index, shape, desired):
    assert_flat_equal(canonicalize_index(index, shape), desired)


ONE_D_OBJS = [
    param(lambda x: x, id="list"),
    param(lambda x: (tuple(x),), id="tuple"),
    param(lambda x: np.array(x), id="array"),
]


@pytest.mark.parametrize("object", ONE_D_OBJS)
@pytest.mark.parametrize(
    ["index", "desired", "expectation"],
    [
        param([1, 3], (10,), (np.array([1, 3]),), does_not_raise(), id="standard"),
        param([3, 1], (10,), (np.array([3, 1]),), does_not_raise(), id="out of order"),
        param(
            [3, 7, 1],
            (10,),
            (np.array([3, 7, 1]),),
            does_not_raise(),
            id="more out of order",
        ),
        param([0, 1, 3], (10,), (np.array([0, 1, 3]),), does_not_raise(), id="low"),
        param([6, 8, 9], (10,), (np.array([6, 8, 9]),), does_not_raise(), id="high"),
        param([0, 9], (10,), (np.array([0, 9]),), does_not_raise(), id="edges"),
        param([9, 10], (10,), None, raises(IndexError), id="just too high"),
        param([5, 30], (10,), None, raises(IndexError), id="too high"),
        param([-7, -3], (10,), (np.array([3, 7]),), does_not_raise(), id="negative"),
        param([3, -1], (10,), (np.array([3, 9]),), does_not_raise(), id="negative low"),
        param(
            [3, -10], (10,), (np.array([3, 0]),), does_not_raise(), id="negative high"
        ),
        param(
            [-10, -1], (10,), (np.array([0, 9]),), does_not_raise(), id="negative edges"
        ),
        param([-11, -15], (10,), None, raises(IndexError), id="negative too high"),
        param([-20, -15], None, raises(IndexError), id="negative too high"),
        param([0.5, 1.5], None, raises(IndexError), id="float"),
        param(["T", "S"], None, raises(IndexError), id="string"),
    ],
)
def test_multiindex_first_axis(object, index, desired, expectation):
    with expectation:
        assert_flat_equal(canonicalize_index(object(index), (10,)), desired)
