#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
""" Tests for jagged io operations. """
import numpy as np
import pytest

from ..api import random
from ..io.npz import load_npz
from ..io.npz import save_npz
from .utils import assert_equal


@pytest.mark.parametrize(
    "compressed", [False, True], ids=["not compressed", "compressed"]
)
def test_roundtrip(compressed, tmp_path):
    path = tmp_path / "jagged.npz"
    saved = random(shape=(20, 20, 20))
    save_npz(path, saved, compressed=compressed)
    loaded = load_npz(path)
    assert_equal(saved, loaded)


def test_wrong_format(tmp_path):
    path = tmp_path / "nonjagged.npz"
    nonjagged = np.random.random((10, 10))
    np.savez(path, nonjagged)
    with pytest.raises(RuntimeError):
        load_npz(path)
