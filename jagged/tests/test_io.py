#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

import pytest
import numpy as np

import jagged
from ..io.npz import save_npz, load_npz


@pytest.mark.parametrize("compressed", [False, True])
def test_roundtrip(compressed, tmp_path):
    path = tmp_path / "jagged.npz"
    saved = jagged.random(shape=[[20, 20, 20]])
    save_npz(path, saved, compressed=compressed)
    loaded = load_npz(path)
    assert saved == loaded


def test_wrong_format(tmp_path):
    path = tmp_path / "nonjagged.npz"
    nonjagged = np.random.random((10, 10))
    np.savez(path, nonjagged)
    with pytest.raises(RuntimeError):
        load_npz(path)
