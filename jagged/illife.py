#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.illife
~~~~~~~~~~~~~

Support for converting jagged arrays to and from Illife vectors.
"""
import numpy as np

from .core import JaggedArray


def illife_to_jagged(ivec: np.ndarray):
    """ Convert an Illife vector to a jagged array. """

    ivec = np.array([np.asarray(arr) for arr in ivec])
    return JaggedArray(
        np.concatenate([arr.flatten() for arr in ivec]),
        shapes=np.asarray([arr.shape for arr in ivec]),
    )


def jagged_to_illife(jarray, copy=True):
    """ Convert a jagged array to an illife vector. """

    return np.array([arr for arr in jarray])
