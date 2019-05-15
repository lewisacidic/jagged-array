#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged utilities. """
import numpy as np
import pytest

from ..utils import is_iterable
from ..utils import sanitize_shape


@pytest.mark.parametrize(
    ["shape", "expected"],
    [
        ((2, (1, 2)), (2, (1, 2))),
        ([2, [1, 2]], (2, (1, 2))),
        (np.array([2, [1, 2]]), (2, (1, 2))),
        ((2, [1, 2]), (2, (1, 2))),
        ([2, (1, 2)], (2, (1, 2))),
        ((2, np.array([1, 2])), (2, (1, 2))),
    ],
)
def test_sanitize_shape(shape, expected):
    result = sanitize_shape(shape)
    assert result == expected
    assert all(type(d) is int for dim in result if is_iterable(dim) for d in dim)
