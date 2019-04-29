#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

from numbers import Integral
from typing import Tuple

import numpy as np

from .jagged import JaggedArray


def random(shape: Tuple[int], random_state=None, data_rvs=None):
    """ Generate a random jagged array. 
    
    Args:
        shape: if 1D, the maximal bounds of the jagged array, otherwise the shape
    
    """
    # get random state
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, Integral):
        random_state = np.random.RandomState(random_state)
    if data_rvs is None:
        data_rvs = random_state.rand

    # get shape
    shape = np.asarray(shape)
    if shape.ndim == 1:
        # max dimensions
        dim_1, *other_dims = shape
        shape = np.stack([random_state.randint(1, dim + 1, dim_1) for dim in other_dims])
    elif shape.dim != 2:
        raise RuntimeError("Parameter {!r} for shape not recognised".format(shape))

    size = shape.prod(axis=0).sum()

    return JaggedArray(data_rvs(size), shape)
