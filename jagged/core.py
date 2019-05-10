#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
"""
jagged.core
~~~~~~~~~~~

Implementation for `JaggedArray`, the core data structure of jagged array.
"""
from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np

from .indexing import getitem
from .typing import ArrayLike
from .typing import AxisLike
from .typing import DtypeLike
from .typing import JaggedShapeLike
from .typing import Number
from .typing import ShapeLike
from .utils import shape_to_shapes
from .utils import shapes_to_shape
from .utils import shapes_to_size


class JaggedArray(np.lib.mixins.NDArrayOperatorsMixin):
    """ Object supporting arrays with jagged axes off an inducing axis.

    Args:
        data:
            The data to be represented flattened as a one dimensional array.
        shape:
            The shape of the data.

    Examples:
        Instantiating a jagged array:
        >>> JaggedArray(np.arange(8), shape=(3, (3, 2, 3)))
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        Using `shapes`:
        >>> JaggedArray(np.arange(8), shapes=[[3], [2], [3]])
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        Using an Illife vector:
        >>> JaggedArray.from_illife([[0, 1, 2], [3, 4], [5, 6, 7]])
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        Using a masked array:
        >>> ma = np.masked.maskedarray(np.arange(8),
        ...                            [[False, False, False],
        ...                             [False, False,  True],
        ...                             [False, False, False]])
        >>> JaggedArray.from_masked(ma)
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        Higher dimensions:
        >>> JaggedArray(np.arange(18), shape=(3, (2, 2, 4), (2, 3, 2)))
        JaggedArray([[[ 0,  1],
                      [ 2,  3]],

                     [[ 4,  5,  6],
                      [ 7,  8,  9]],

                     [[10, 11],
                      [12, 13],
                      [14, 15],
                      [16, 17]])

        >>> JaggedArray(np.arange(18), shapes=[[4, 2], [2, 3], [2, 2]])
        JaggedArray([[[ 0,  1],
                      [ 2,  3]],

                     [[ 4,  5,  6],
                      [ 7,  8,  9]],

                     [[10, 11],
                      [12, 13],
                      [14, 15],
                      [16, 17]])
     """

    __array_priority__ = 42

    def __init__(
        self,
        data: ArrayLike,
        shape: Optional[JaggedShapeLike] = None,
        shapes: Optional[np.ndarray] = None,
    ) -> JaggedArray:
        """ Initialize a jagged array.

        Please see `help(JaggedArray)` for more info. """

        if shape is None:
            if shapes is None:
                raise ValueError("Either `shape` or `shapes` must be passed.")
            else:
                self.shape = shapes_to_shape(shapes)
        else:
            if shapes is None:
                self.shape = shape
            else:
                raise ValueError(
                    "`shape` and `shapes` cannot be passed simultaneously."
                )

        self.data = data
        self._verify_consistency()

    def _verify_consistency(self):
        """ Check that the data fits the stated size """
        shape_size = shapes_to_size(self.shapes)
        if not self.data.size == shape_size:
            msg = f"Size of data ({self.data.size}) does not match the size of shape ({shape_size})"
            raise ValueError(msg)

    def __getstate__(self):
        """ returns a tuple to allow easy pickling """
        return self.data.tolist(), self.shape

    def __setstate__(self, state):
        """ initializes class with a state to allow recovery from pickling """
        self.data = state.data
        self.shape = state.shape

    def __len__(self):
        """ Get the length of the jagged array.

        This is the size along the inducing dimension.

        Examples:
            >>> len(JaggedArray(np.arange(8), (3, (3, 2, 3))))
            3

            >>> len(JaggedArray(np.arange(10), (5, (1, 2, 3, 2, 2))))
            5
        """
        return self.shape[0]

    def __sizeof__(self):
        return self.nbytes

    __getitem__ = getitem

    def __str__(self) -> str:
        raise NotImplementedError

    __repr__ = __str__

    @property
    def data(self) -> np.ndarray:
        """ 1D array storing all entries of the  array.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).data
            array([0, 1, 2, 3, 4, 5, 6, 7])

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).data
            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                    17])
        """
        return self._data

    @data.setter
    def data(self, value: ArrayLike):
        value = np.asarray(value)
        if value.ndim > 1:
            raise ValueError(f"`data` must be one dimensional.  Was ({value.ndim}).")
        self._data = value

    @property
    def shape(self) -> Tuple[int]:
        """ Tuple of array dimensions.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).shape
            (3, (3, 2, 3))

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).shape
            (3, (4, 2, 2), (2, 3, 2))

        See Also:
            JaggedArray.shapes
        """
        return self._shape

    @shape.setter
    def shape(self, shape: JaggedShapeLike):
        self._shape = tuple(ax if isinstance(ax, int) else tuple(ax) for ax in shape)

    @property
    def shapes(self) -> np.ndarray:
        """ The shapes of the arrays along the inducing axis.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).shapes
            array([[3],
                   [2],
                   [3]])

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).shapes
            array([[4, 2],
                   [2, 3],
                   [2, 2]])
        """
        return shape_to_shapes(self.shape)

    @shapes.setter
    def shapes(self, value: np.ndarray):
        self.shape = shapes_to_shape(value)

    @property
    def size(self) -> int:
        """ the number of elements in the jagged array.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).size
            8

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).size
            18
        """
        return self.data.size

    @property
    def sizes(self) -> Tuple[int]:
        """ the sizes of the subarrays along the inducing axis.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).sizes
            (3, 2, 3)

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).sizes
            (8, 6, 4)
        """
        return self.shapes.prod(axis=1)

    @property
    def nbytes(self) -> int:
        """ the number of bytes taken up by the jagged array.

        Note:
            As an implementation detail, this does not factor in cached data.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).nbytes
            110

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).nbytes
            224
        """
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        """ the number of dimensions.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).ndim
            2

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).ndim
            3
        """
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """ the dtype of the contained data.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).dtype
            dtype('int64')

            >>> JaggedArray(np.arange(8, dtype='f4'), (3, (3, 2, 3))).dtype
            dtype('float32')
        """
        return self.data.dtype

    @property
    def limits(self) -> np.ndarray:
        """ the shape of the 'convex hull' of the array.

        This would be the shape of the resultant dense array.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).limits
            (3, 3)

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).limits
            (3, 4, 3)

        See Also:
            JaggedArray.shape
        """
        return tuple(ax if isinstance(ax, int) else max(ax) for ax in self.shape)

    @property
    def jagged_axes(self) -> Tuple[bool]:
        """ The indexes of jagged axes

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).jagged_axes
            (1,)

            >>> JaggedArray(np.arange(16), (3, (3, 2, 3), 2)).jagged_axes
            (1,)
         """
        return tuple(i for i, ax in enumerate(self.shape) if isinstance(ax, tuple))

    def copy(self) -> JaggedArray:
        """ copy the jagged array.

        Examples:
            >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
            >>> ja2 = ja.copy()
            >>> jagged.array_equal(ja == ja2)
            True

            Data is copied:
            >>> ja2[...] = 42
            >>> ja2
            JaggedArray([[42, 42, 42],
                         [42, 42],
                         [42, 42, 42]])

            >>> ja
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5, 6, 7]])
        """
        # tuple is immutable, so this is fine to pass without copy
        return JaggedArray(self.data.copy(), shape=self.shape)

    def astype(self, dtype: DtypeLike) -> JaggedArray:
        """ a copy of the array with the data as a given data type.

        Args:
            dtype:
                the numpy dtype to use to represent data.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).astype('i4')
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5, 6, 7]], dtype=int32)

            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).astype('f2')
            JaggedArray([[0., 1., 2.],
                         [3., 4.],
                         [5., 6., 7.]], dtype=float16)
        """
        res = self.copy()
        res.data = self.data.astype(dtype)
        return res

    @classmethod
    def from_illife(cls, arr: ArrayLike) -> JaggedArray:
        """ Create a jagged array from an Illife vector (array of arrays).

        Args:
            arr:
                Illife vector to convert.

        Examples:
            >>> JaggedArray.from_illife([[0, 1, 2],
            ...                          [3, 4],
            ...                          [5, 6, 7]])
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5, 6, 7]])

            >>> JaggedArray.from_illife([[[0, 1, 2],
            ...                           [3, 4, 5]],
            ...                          [[6, 7]],
            ...                          [[ 8],
            ...                           [ 9],
            ...                           [10]],
            ...                          [[11]]])
            JaggedArray([[[0, 1, 2],
                          [3, 4, 5]],

                         [[6, 7]],

                         [[8]],

                         [[9]],

                         [[10]],

                         [[11]]])
        """
        from .illife import from_illife

        return from_illife(arr)

    @classmethod
    def from_masked(cls, arr: np.masked.masked_array) -> JaggedArray:
        """ Create a jagged array from a masked numpy array.

        Args:
            arr:
                Masked numpy array to convert.

        Examples:
            >>> arr = np.ma.masked_array(np.array([[0, 1, 2],
            ...                                    [3, 4, 0],
            ...                                    [5, 0, 0],
            ...                                    [6, 7, 8]]),
            ...                          np.array([[False, False, False],
            ...                                    [False, False,  True],
            ...                                    [False,  True,  True],
            ...                                    [False, False, False]]))
            >>> JaggedArray.from_masked(arr)
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5],
                         [6, 7, 8]])

        Notes:
            The first masked value in a given direction is assumed to be the
            end of the array.
        """
        from .masked import from_masked

        return from_masked(arr)

    @classmethod
    def from_array(
        cls, arr: np.ndarray, masked_value: Optional[Any] = None
    ) -> JaggedArray:
        """ Create a jagged array from a (full) array with a masked value.

        Args:
            arr:
                array to convert.

            masked_value:
                The masked value.  If no value is passed and the array is
                compatible with float, this will be `nan`, otherwise `None`.

        Examples:
            >>> arr = np.array([[     0.,     1.,     2.],
            ...                 [     3.,     4., np.nan],
            ...                 [     5., np.nan, np.nan],
            ...                 [     6.,     7.,     8.]])
            >>> JaggedArray.from_array(arr).astype(int)
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5],
                         [6, 7, 8]])
        """
        raise NotImplementedError

    def to_masked(self) -> np.mask.masked_array:
        """ convert the array to a dense masked array.

        Examples:
            >>> JaggedArray(np.arange(8), [[3, 2, 3]]).to_masked()
            masked_array(data =
             [[0 1 2]
             [3 4 --]
             [5 6 7]],
                         mask =
             [[False False False]
             [False False  True]
             [False False False]],
                   fill_value = 999999)

            >>> JaggedArray(np.arange(33), np.array([[3, 2, 3],
            ...                                      [3, 6, 4]])).to_masked()
            masked_array(data =
             [[[0 1 2 -- -- --]
              [3 4 5 -- -- --]
              [6 7 8 -- -- --]]
             [[9 10 11 12 13 14]
              [15 16 17 18 19 20]
              [-- -- -- -- -- --]]
             [[21 22 23 24 -- --]
              [25 26 27 28 -- --]
              [29 30 31 32 -- --]]],
                         mask =
             [[[False False False  True  True  True]
              [False False False  True  True  True]
              [False False False  True  True  True]]
             [[False False False False False False]
              [False False False False False False]
              [ True  True  True  True  True  True]]
             [[False False False False  True  True]
              [False False False False  True  True]
              [False False False False  True  True]]],
                   fill_value = 999999)
        """
        from jagged.masked import to_masked

        return to_masked(self)

    def to_illife(self) -> np.ndarray:
        """ Convert the jagged array to an Illife vector.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).to_illife()
            array([array([0, 1, 2]),
                   array([3, 4]),
                   array([5, 6, 7])], dtype=object)

            >>> JaggedArray(np.arange(33), (3, (3, 2, 3), (3, 6, 4))).to_illife()
            array([array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8]]),
                   array([[ 9, 10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19, 20]]),
                   array([[21, 22, 23, 24],
                          [25, 26, 27, 28],
                          [29, 30, 31, 32]])], dtype=object)
        """
        from .illife import to_illife

        return to_illife(self)

    def to_array(self, fill_value: Optional[Any] = None) -> np.ndarray:
        """ Convert to a dense array.

        Args:
            fill_value:
                The value to fill in the array.  If `None` (as default) and the
                array can be converted to floats, we will use `np.nan`.

        Notes:
            Using np.nan as the `fill_value` will cause a jagged array of integer dtype
            to be coerced into a float array due to the lack of a numpy nan integer type.

        Examples:
            >>> JaggedArray(np.arange(8),  (3, (3, 2, 3)))).to_array()
            array([array([  0.,   1.,   2.]),
                   array([  3.,   4.,  nan]),
                   array([  5.,   6.,   7.])], dtype=np.float64)
        """
        raise NotImplementedError

    def clip(self, a_min=Optional[Number], a_max=Optional[Number]):
        """ Clip the values of the array.

        Args:
            a_min:
                Lower bound of clipping interval.  Values below this will be
                set to this value.
            a_max:
                Upper bound of clipping interval.  Values above this will be
                set to this value.

        Examples:
            >>> JaggedArray(np.arange(5), (3, (2, 1, 2))).clip(a_min=2)
            JaggedArray([[2, 2],
                         [2],
                         [3, 4]])

            >>> JaggedArray(np.arange(5), (3, (2, 1, 2))).clip(a_max=2)
            JaggedArray([[0, 1],
                         [2],
                         [2, 2]])

            >>> JaggedArray(np.arange(5), (3, (2, 1, 2))).clip(a_min=1, a_max=3)
            JaggedArray([[1, 1],
                         [2],
                         [3, 3]])
        """
        raise NotImplementedError

    def conjugate(self) -> JaggedArray:
        """ Return the element-wise complex conjugate.

        The complex conjugate of a number is obtained by changing the sign of
        its imaginary part.

        Returns:
            JaggedArray

        Examples:
            >>> JaggedArray([1j, 1 + j, 1 - j], (2, (2, 1))).conjugate()
            JaggedArray([[0.-1.j, 1.-1.j],
                         [1.+j]], dtype=complex128)
        """
        raise NotImplementedError

    conj = conjugate

    def fill(self, value: Any):
        """ Fill the jagged array with a scalar value.

        Args:
            value (any):
                All elements of `a` will be assigned this value.

        Examples:
            >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
            >>> ja.fill(0)
            >>> ja
            JaggedArray([[0, 0, 0],
                         [0, 0],
                         [0, 0, 0]])
        """

        self.data[...] = value

    @property
    def flat(self) -> np.ndarray:
        """ Return an iterator over the entries of the jagged array.

        Examples:
            >>> list(JaggedArray(np.arange(8), (3, (3, 2, 3))).flat)
            [0, 1, 2, 3, 4, 5, 6, 7]

        See also:
            ndarray.flat
            numpy.flatiter
        """
        return self.data.flat

    def flatten(self) -> np.ndarray:
        """ Flatten the jagged array.

        This creates a **copy** of the data.

        Examples:
            >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
            >>> flattened = ja.flatten()
            >>> flattened
            array([0, 1, 2, 3, 4, 5, 6])
            >>> flattened[...] = 0
            >>> ja
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5, 6, 7]])

        See Also:
            JaggedArray.ravel
            jagged.flatten
            jagged.ravel
        """
        return self.data.copy()

    def ravel(self) -> np.ndarray:
        """ Ravel the array.

        Creates a **view** of the data.

        Examples:
            >>> ja = JaggedArray(np.arange(7), (3, (3, 2, 3)))
            >>> ravelled = ja.ravel()
            >>> ravelled
            array([0, 1, 2, 3, 4, 5, 6])
            >>> ravelled[...] = 0
            >>> ja
            JaggedArray([[0, 0, 0],
                         [0, 0],
                         [0, 0, 0]])

        See Also:
            JaggedArray.ravel
            jagged.flatten
            jagged.ravel
        """
        return self.data

    @property
    def imag(self) -> JaggedArray:
        """ Get the imaginary part of the jagged array.

        Examples:
            >>> JaggedArray([1j, 1 + 1j, 1 - 1j], (2, (2, 1)))
            JaggedArray([[1, 1],
                         [-1]])
        """
        raise NotImplementedError

    @imag.setter
    def imag(self, values):
        raise NotImplementedError

    @property
    def real(self):
        """ Get the real part of the jagged array.

        Examples:
            >>> JaggedArray([1j, 1 + 1j, 1 - 1j], (2, (2, 1)))
            JaggedArray([[0, 1],
                         [1]])
        """
        raise NotImplementedError

    @real.setter
    def real(self, values):
        raise NotImplementedError

    def reshape(self, shape: ShapeLike) -> JaggedArray:
        """ Reshape the array.

        Args:
            shape:
                the shape with which to reorient the data.

        Examples:
            >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
            >>> ja.reshape([[2, 3, 3]])
            JaggedArray([[0, 1],
                         [2, 3, 4],
                         [5, 6, 7]])
            >>> ja.reshape([[3, 3, 3]])
            ValueError: total size of new array must be unchanged.
        """
        return NotImplementedError

    def resize(self, shape: ShapeLike) -> JaggedArray:
        """ resize the arrays """

        return NotImplementedError

    def squeeze(self, axis: Optional[AxisLike] = None):
        """ Squeeze the given axis.

        This will remove axes from shape with only single dimensional entries.

        Args:
            axis:
                the axes to squeeze.

        See also:
            :func:`squeeze`: equivalent standalone function
        """
        from .api import squeeze

        return squeeze()

    def expand_dims(self, axis: int = -1) -> JaggedArray:
        """ Expand dimensions.

        Args:
            axis:
                the axis after which to insert the dimension

        See also:
            :func:`expand_dims`: equivalent standalone function
        """
        from .api import expand_dims

        return expand_dims(self, axis=axis)

    def digitize(self, bins: ArrayLike, right: bool = False) -> JaggedArray:
        """ Return the indices of the bins for each value in array.

        Args:
            bins:
                Array of 1-dimensional, monotonic bins.

            right:
                Whether the intervals include the right or the left bin edge.
        """
        return self.__class__(np.digitize(self.data, bins, right=right), self.shape)

    def trace(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: Optional[DtypeLike] = None,
        out: Optional[np.ndarray] = None,
    ):
        """ Return the sum along diagonals of a jagged array.

        Args:
            offset:
                Offset of the diagonal from the main diagonal. Can be both positive and
                negative to access upper and lower triangle respectively.

            axis1, axis2:
                Axes to be used as the first and second axis of the subarrays from
                which the diagonals should be taken.

            dtype:
                The data-type of the returned array.

            out:
                The array in which the output is placed.
        """

        raise NotImplementedError()
