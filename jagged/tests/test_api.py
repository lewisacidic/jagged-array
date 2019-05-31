#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged api """
import numpy as np
import pytest
from pytest import param

from ..api import arange
from ..api import concatenate
from ..core import JaggedArray
from .utils import assert_equal
from .utils import does_not_raise
from .utils import raises


@pytest.mark.parametrize(
    ["arrs", "desired", "axis", "expectation"],
    [
        param(
            [arange(shape=(2, (1, 2))), arange(shape=(2, (1, 2)))],
            JaggedArray.from_iliffe([[0], [1, 2], [0], [1, 2]]),
            0,
            does_not_raise(),
            id="concat same shapes along inducing axis",
        ),
        param(
            [arange(shape=(2, (1, 2))), arange(shape=(3, (1, 2, 1)))],
            JaggedArray.from_iliffe([[0], [1, 2], [0], [1, 2], [3]]),
            0,
            does_not_raise(),
            id="concat different shapes along inducing axis",
        ),
        param(
            [
                arange(shape=(2, (1, 2))),
                arange(shape=(3, (2, 1, 2))),
                arange(shape=(2, (1, 3))),
            ],
            JaggedArray.from_iliffe([[0], [1, 2], [0, 1], [2], [3, 4], [0], [1, 2, 3]]),
            0,
            does_not_raise(),
            id="concat three different shapes along inducing axis",
        ),
        param(
            [
                arange(shape=(2, (1, 2)), dtype="i4"),
                arange(shape=(2, (2, 1)), dtype="i2"),
            ],
            JaggedArray.from_iliffe([[0], [1, 2], [0, 1], [2]], dtype="i4"),
            0,
            does_not_raise(),
            id="concat different dtypes along inducing axis",
        ),
        param(
            [arange(shape=(2, (1, 2))), [[0, 1], [2]]],
            JaggedArray.from_iliffe([[0], [1, 2], [0, 1], [2]]),
            0,
            does_not_raise(),
            marks=pytest.mark.xfail,
            id="concat iliffe-like along inducing axis",
        ),
        param(
            [arange(shape=(2, (1, 2))), np.arange(4).reshape(2, 2)],
            JaggedArray.from_iliffe([[0], [1, 2], [0, 1], [2, 3]]),
            0,
            does_not_raise(),
            marks=pytest.mark.xfail,
            id="concat numpy array along inducing axis",
        ),
        param(
            [
                arange(shape=(2, (1, 2))),
                np.ma.masked_array(np.arange(4), [[False, False], [False, True]]),
            ],
            JaggedArray.from_iliffe([[0], [1, 2], [0, 1], [2]]),
            0,
            does_not_raise(),
            marks=pytest.mark.xfail,
            id="concat masked array along inducing axis",
        ),
        param(
            [arange(shape=(2, (1, 2))), arange(shape=(2, (1, 2), 1))],
            None,
            0,
            raises(ValueError),
            id="concat different ndims along inducing axis",
        ),
        param(
            [arange(shape=(2, (2, 1))), arange(shape=(2, (2, 1)))],
            JaggedArray.from_iliffe([[0, 1, 0, 1], [2, 2]]),
            1,
            does_not_raise(),
            id="concat same shapes along jagged axis",
        ),
        param(
            [arange(shape=(2, (1, 3))), arange(shape=(2, (2, 1)))],
            JaggedArray.from_iliffe([[0, 0, 1], [1, 2, 3, 2]]),
            1,
            does_not_raise(),
            id="concat different shapes along jagged axis",
        ),
        param(
            [
                arange(shape=(2, (1, 2))),
                arange(shape=(2, (2, 1))),
                arange(shape=(2, (3, 1))),
            ],
            JaggedArray.from_iliffe([[0, 0, 1, 0, 1, 2], [1, 2, 2, 3]]),
            1,
            does_not_raise(),
            id="concat three along jagged axis",
        ),
        param(
            [arange(shape=(2, (1, 2))), np.arange(4).reshape(2, 2)],
            JaggedArray.from_iliffe([[0, 0, 1], [1, 2, 2, 3]]),
            1,
            does_not_raise(),
            id="concat numpy array along jagged axis",
        ),
        param(
            [
                arange(shape=(2, (2, 1))),
                np.ma.masked_array(np.arange(4), [[False, False], [False, True]]),
            ],
            JaggedArray.from_iliffe([[0, 1, 0, 1], [2, 2]]),
            1,
            does_not_raise(),
            marks=pytest.mark.xfail,
            id="concat masked array along jagged axis",
        ),
        param(
            [arange(shape=(2, (1, 2))), [[0], [1, 2]]],
            JaggedArray.from_iliffe([[0, 0], [1, 2, 1, 2]]),
            1,
            does_not_raise(),
            marks=pytest.mark.xfail,
            id="concat iliffe-like along jagged axis",
        ),
        param(
            [arange(shape=(3, (1, 2, 3))), arange(shape=(2, (1, 2)))],
            None,
            1,
            raises(ValueError),
            id="concat mismatched shapes along jagged axis",
        ),
        param(
            [arange(shape=(2, (1, 2), 2)), arange(shape=(2, (1, 2), 1))],
            None,
            1,
            raises(ValueError),
            id="concat mismatched shapes in axis 2 along jagged axis",
        ),
        param(
            [arange(shape=(2, (1, 2), 2)), arange(shape=(2, (1, 2), 1))],
            JaggedArray.from_iliffe([[[0, 1, 0]], [[2, 3, 1], [4, 5, 2]]]),
            2,
            does_not_raise(),
            id="concat along flat dim",
        ),
        param(
            [arange(shape=(2, (2, 1), 2)), arange(shape=(2, (1, 2), 1))],
            JaggedArray.from_iliffe([[[0, 1, 0]], [[2, 3, 1], [4, 5, 2]]]),
            2,
            raises(ValueError),
            id="concat along flat dim with mismatched jagged dim",
        ),
    ],
)
def test_concatenate(arrs, desired, axis, expectation):
    with expectation:
        result = concatenate(arrs, axis=axis)
        assert_equal(result, desired)
        assert result.dtype == desired.dtype
