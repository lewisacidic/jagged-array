#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged arrays. """
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest
from pytest import raises

import jagged as jgd
from jagged import JaggedArray


dtypes = "f8", "f4", "i8", "i4"


@pytest.mark.parametrize(
    ("kwargs", "expectation"),
    [
        ({"data": [1, 2, 3], "shape": (2, (1, 2))}, does_not_raise()),
        ({"data": (1, 2, 3), "shape": (2, (1, 2))}, does_not_raise()),
        ({"data": np.array([1, 2, 3]), "shape": (2, (1, 2))}, does_not_raise()),
        ({"data": [1, 2, 3], "shape": [2, [1, 2]]}, does_not_raise()),
        ({"data": [1, 2, 3], "shapes": [[1], [2]]}, does_not_raise()),
        ({"data": [1, 2, 3], "shapes": np.array([[1], [2]])}, does_not_raise()),
        ({"data": [1, 2, 3]}, raises(ValueError)),
        (
            {"data": [1, 2, 3], "shapes": [[1], [2]], "shape": (2, (1, 2))},
            raises(ValueError),
        ),
    ],
    ids=[
        "data as list",
        "data as tuple",
        "data as array",
        "shape as lists",
        "shapes as lists",
        "shapes as array",
        "no shape or shapes",
        "both shape and shapes",
    ],
)
def test_instantiation(kwargs, expectation):
    with expectation:
        ja = JaggedArray(**kwargs)
        assert isinstance(ja.data, np.ndarray)
        assert isinstance(ja.shape, tuple)
        assert ja.shape == (2, (1, 2))


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


@pytest.mark.skip(reason="not yet implemented")
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
        (0, raises(RuntimeError)),
        (1, does_not_raise()),
        (2, does_not_raise()),
        ((0, 1), raises(RuntimeError)),
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
