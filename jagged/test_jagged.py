#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

""" tests for jagged arrays. """

import pytest
import numpy as np

from .jagged import JaggedArray


@pytest.fixture(name="ja")
def jagged_array():
    return JaggedArray(np.arange(8), [[3, 2, 3]])


@pytest.fixture(name="jab")
def big_jagged_array():
    return JaggedArray(np.arange(33), np.array([[3, 2, 3], [3, 6, 4]]))


class TestVerify(object):
    def test_data_dimensions(self):
        with pytest.raises(ValueError):
            JaggedArray(np.arange(8).reshape(4, 2), [[3, 2, 3]])

    def test_shape_dimensions(self):
        with pytest.raises(ValueError):
            JaggedArray(np.arange(8), [3, 2, 3])

    def test_shape_dimensions_too_high(self):
        with pytest.raises(ValueError):
            JaggedArray(np.arange(8), [[[3, 2, 3]]])

    def test_shape_not_equal(self):
        with pytest.raises(ValueError):
            JaggedArray(np.arange(9), [[3, 2, 3]])


def test_eq(ja):
    assert ja == JaggedArray(np.arange(8), [[3, 2, 3]])
    assert ja != JaggedArray(np.arange(8) + 1, [[3, 2, 3]])
    assert ja != JaggedArray(np.arange(8), [[3, 3, 2]])


def test_copy(ja):
    assert ja == ja.copy()


def test_as_type(ja):
    assert ja.astype(float).data.dtype == float


def test_get(ja):
    assert np.array_equal(ja[0], [0, 1, 2])


class TestToFrom(object):
    def test_array(self, ja):
        assert ja == JaggedArray.from_aoa(ja.to_aoa())

    def test_masked(self, ja):
        assert ja == JaggedArray.from_masked(ja.to_masked())
