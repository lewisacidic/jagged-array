#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged arrays. """
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest

import jagged as jgd


dtypes = "f8", "f4", "i8", "i4"


@pytest.fixture
def jagged(limits, dtype):
    if np.issubtype(dtype, np.integer):

        def data_rvs(n):
            return np.random.randint(-1000, 1000, n)

    else:
        data_rvs = None

    return jgd.random(limits, data_rvs=data_rvs).astype(dtype)


@pytest.fixture
def masked(jagged):
    return jagged.to_masked()


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "reduction,kwargs",
    [
        ("sum", {}),
        ("sum", {"dtype": np.float32}),
        ("mean", {}),
        ("mean", {"dtype": np.float32}),
        ("prod", {}),
        ("max", {}),
        ("min", {}),
        ("std", {}),
        ("var", {}),
    ],
)
@pytest.mark.parametrize(
    "axis,expectation",
    [
        (None, does_not_raise()),
        (0, pytest.raises(RuntimeError)),
        (1, does_not_raise()),
        (2, does_not_raise()),
        ((0, 1), pytest.raises(RuntimeError)),
        ((1, 2), does_not_raise()),
        (-1, does_not_raise()),
        ((1, -1), does_not_raise()),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_reductions(jagged, masked, reduction, axis, keepdims, kwargs, expectation):
    with expectation:
        jgd_res = getattr(jagged, reduction)(axis=axis, keepdims=keepdims)
        msk_res = getattr(masked, reduction)(axis=axis, keepdims=keepdims)
        assert jgd_res == msk_res
