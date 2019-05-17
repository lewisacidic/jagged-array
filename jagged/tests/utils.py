#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Utility functions for writing tests for jagged-array. """
import numpy as np

from ..api import array_equal


def assert_equal(jarr1, jarr2):
    assert array_equal(jarr1, jarr2)


def assert_illife_equal(ivec1, ivec2):
    assert all(np.array_equal(a1, a2) for a1, a2 in zip(ivec1, ivec2))
