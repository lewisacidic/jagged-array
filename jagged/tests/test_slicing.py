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


@pytest.mark.parametrize(
    "object",
    [
        param(lambda x: x, id="integer"),
        param(lambda x: [x], id="list"),
        param(lambda x: ((x,),), id="tuple"),
        param(lambda x: np.array(x), id="array"),
    ],
)
@pytest.mark.parametrize(
    ["index", "desired", "expectation"],
    [
        param(5, (5,), does_not_raise(), id="standard"),
        param(0, (0,), does_not_raise(), id="low"),
        param(9, (9,), does_not_raise(), id="high"),
        param(-3, (7,), does_not_raise(), id="negative"),
        param(-1, (9,), does_not_raise(), id="negative low"),
        param(-10, (0,), does_not_raise(), id="negative high"),
        param(1.5, None, raises(IndexError), id="float"),
        param("c", None, raises(IndexError), id="string"),
        param(10, None, raises(IndexError), id="just too high"),
        param(20, None, raises(IndexError), id="too high"),
        param(-11, None, raises(IndexError), id="just too low"),
        param(-21, None, raises(IndexError), id="too low"),
    ],
)
def test_zero_d(object, index, desired, expectation):
    with expectation:
        assert_flat_equal(canonicalize_index(object(index), (10,)), desired)


@pytest.mark.parametrize(
    "object",
    [
        param(lambda x: x, id="list"),
        param(lambda x: (tuple(x),), id="tuple"),
        param(lambda x: np.array(x), id="array"),
    ],
)
@pytest.mark.parametrize(
    ["index", "desired", "expectation"],
    [
        param([1, 3], (np.array([1, 3]),), does_not_raise(), id="standard"),
        param([3, 1], (np.array([3, 1]),), does_not_raise(), id="out of order"),
        param(
            [3, 7, 1], (np.array([3, 7, 1]),), does_not_raise(), id="more out of order"
        ),
        param([0, 1, 3], (np.array([0, 1, 3]),), does_not_raise(), id="low"),
        param([6, 8, 9], (np.array([6, 8, 9]),), does_not_raise(), id="high"),
        param([0, 9], (np.array([0, 9]),), does_not_raise(), id="edges"),
        param([9, 10], None, raises(IndexError), id="just too high"),
        param([5, 30], None, raises(IndexError), id="too high"),
        param([-7, -3], (np.array([3, 7]),), does_not_raise(), id="negative"),
        param([3, -1], (np.array([3, 9]),), does_not_raise(), id="negative low"),
        param([3, -10], (np.array([3, 0]),), does_not_raise(), id="negative high"),
        param([-10, -1], (np.array([0, 9]),), does_not_raise(), id="negative edges"),
        param([-11, -15], None, raises(IndexError), id="negative too high"),
        param([-20, -15], None, raises(IndexError), id="negative too high"),
        param([0.5, 1.5], None, raises(IndexError), id="float"),
        param(["T", "S"], None, raises(IndexError), id="string"),
    ],
)
def test_one_d(object, index, desired, expectation):
    with expectation:
        assert_flat_equal(canonicalize_index(object(index), (10,)), desired)
