#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for ufuncs for jagged arrays. """
from operator import add
from operator import gt
from operator import iadd
from operator import ifloordiv
from operator import imul
from operator import isub
from operator import itruediv
from operator import lt
from operator import mul
from operator import ne
from operator import sub

import pytest
from numpy.testing import assert_equal

import jagged

operators = mul, add, sub, gt, lt, ne
operators_inplace = iadd, isub, imul, itruediv, ifloordiv

shapes = ((2, 2, 3)), ((3, 2, 3), (2, 4, 2))


@pytest.mark.skip(reason="not yet implemented")
@pytest.mark.parametrize("func", operators)
@pytest.mark.parametrize("shape", shapes)
def test_elementwise_binary(func, shape):
    xj = jagged.random(shape)
    yj = jagged.random(shape)
    x = xj.to_array()
    y = yj.to_array()
    assert_equal(func(xj, yj).to_array(), func(x, y))


@pytest.mark.skip(reason="not yet implemented")
@pytest.mark.parametrize("func", operators_inplace)
@pytest.mark.parametrize("shape", shapes)
def test_elementwise_binary_inplace(func, shape):
    xj, yj = jagged.random(shape), jagged.random(shape)
    x, y = xj.to_array(), yj.to_array()

    func(xj, yj)
    func(x, y)
    assert_equal(xj.to_array(), x)
