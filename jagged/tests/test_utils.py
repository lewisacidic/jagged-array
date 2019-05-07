#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged utilities """
import numpy as np
import pytest

from ..utils import random


@pytest.fixture
def random_shape():
    return np.random.randint(5, 10, 5)


@pytest.fixture
def random_jagged(random_shape):
    return random(random_shape)


class TestRandom(object):
    @pytest.mark.repeat(10)
    def test_shape(self, random_jagged, random_shape):
        assert all(
            shape <= limits for limits, shape in zip(random_jagged.limits, random_shape)
        )

    def test_values(self, random_jagged):
        assert all(0 < random_jagged.data < 1)
