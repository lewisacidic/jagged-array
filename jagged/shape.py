#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.shape
~~~~~~~~~~~~

Jagged shape functionality for jagged arrays.
"""
import warnings
from typing import Tuple

import numpy as np

from .typing import JaggedShapeLike
from .utils import is_integer
from .utils import sanitize_shape
from .utils import shape_to_shapes
from .utils import shape_to_size
from .utils import shapes_to_shape


class JaggedShape(tuple):
    """ A jagged shape.

    Examples:
        >>> ja = JaggedShape((3, (1, 2, 3), (2, 2, 2)); ja
        (3, (1, 2, 3), 2)

        >>> ja = JaggedShape.from_shapes([[1, 2], [2, 2], [3, 2]]); ja
        (3, (1, 2, 3), 2)

        >>> ja.size
        12

        >>> ja.sizes
        (2, 4, 6)

        >>> ja.ndim
        3

        >>> ja.limits
        (3, 3, 2)

        >>> ja.jagged_axes
        (1,)

        >>> ja.to_shapes()
        array([[1, 2],
               [2, 2],
               [3, 2]])
    """

    def __new__(cls, shape: JaggedShapeLike):
        return super(JaggedShape, cls).__new__(cls, sanitize_shape(shape))

    def __init__(self, shape: JaggedShapeLike):
        if not len(self.jagged_axes):
            warnings.warn("Shape is not jagged. Consider using a numpy array.")

    def to_shapes(self):
        """ the indexed shapes along the inducing axis

        Examples:
            >>> JaggedShape((3, (3, 2, 3))).to_shapes()
            array([[3],
                   [2],
                   [3]])

            >>> JaggedShape((3, (4, 2, 2), (2, 3, 2))).to_shapes()
            array([[4, 2],
                   [2, 3],
                   [2, 2]])
         """
        return shape_to_shapes(self)

    @property
    def size(self):
        """ the size of the shape.

        Examples:
            >>> JaggedShape((3, (3, 2, 3))).size
            8

            >>> JaggedShape((3, (4, 2, 2), (2, 3, 2))).size
            18
         """
        return shape_to_size(self)

    @property
    def sizes(self):
        """ the sizes of the subarrays allong the inducing axis

        Examples:
            >>> JaggedShape((3, (3, 2, 3))).sizes
            (3, 2, 3)

            >>> JaggedShape((3, (4, 2, 2), (2, 3, 2))).sizes
            (8, 6, 4)
        """
        return tuple(self.to_shapes().prod(axis=1))

    @property
    def ndim(self):
        """ the number of dimensions indexed over

        Exmaples:
            >>> JaggedShape((3, (3, 2, 3))).ndim
            2

            >>> JaggedShape((3, (4, 2, 2), (2, 3, 2))).ndim
            3

            >>> JaggedShape((3, 2, 3, (3, 2, 2))).ndim
            4
        """
        return len(self)

    @property
    def limits(self):
        """ the length of the largest subarray on each axis

        Examples:
            >>> JaggedShape((3, (3, 2, 3))).limits
            (3, 3)

            >>> JaggedShape((3, (4, 2, 2), (2, 3, 2))).limits
            (3, 4, 3)
        """
        return tuple(ax if is_integer(ax) else max(ax) for ax in self)

    @property
    def jagged_axes(self) -> Tuple[bool]:
        """ The indexes of jagged axes

        Examples:
            >>> JaggedShape((3, (3, 2, 3))).jagged_axes
            (1,)

            >>> JaggedShape((3, (3, 2, 3), 2)).jagged_axes
            (1,)

            >>> JaggedShape((3, (3, 2, 3), 2, (1, 2, 1))).jagged_axes
            (1, 3)

            >>> JaggedShape((3, (3, 2, 3), (1, 1, 1))).jagged_axes
            (1,)
        """
        return tuple(i for i, ax in enumerate(self) if isinstance(ax, tuple))

    @classmethod
    def from_shapes(cls, shapes: np.ndarray):
        """ instantiate a JaggedShape from an array of shapes

        Examples:
            >>> JaggedShape.from_shapes([[3], [2], [3]])
            (3, (3, 2, 3))

            >>> JaggedShape.from_shapes([[3, 2], [2, 2], [3, 2]])
            (3, (3, 2, 3), 2)
        """
        return cls(shapes_to_shape(shapes))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
