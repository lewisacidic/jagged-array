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

from .shape import JaggedShape
from .typing import ArrayLike
from .typing import AxisLike
from .typing import DtypeLike
from .typing import JaggedShapeLike
from .typing import Number
from .typing import ShapeLike
from .utils import jagged_to_string


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
                self.shape = JaggedShape.from_shapes(shapes)
        else:
            if shapes is None:
                self.shape = JaggedShape(shape)
            else:
                msg = "`shape` and `shapes` cannot be passed simultaneously."
                raise ValueError(msg)
        self.data = data
        self._verify_consistency()

    def _verify_consistency(self):
        """ Check that the data fits the stated size. """
        dsize, isize = self.data.size, self.shape.size
        if not dsize == isize:
            msg = f"Size of data ({dsize}) does not match the size of shape ({isize})"
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

    def __str__(self) -> str:
        """ Return a string of the jagged array.

        Examples:
            >>> str(JaggedArray(np.arange(8), (3, (3, 2, 3))))
            [[0 1 2]
             [3 4]
             [5 6 7]]

            >>> str(JaggedArray(np.arange(8), (3, 1, (3, 2, 3))))
            [[[0 1 2]]

             [[3 4]]

             [[5 6 7]]]

            >>> str(JaggedArray(np.arange(8), (3, (3, 2, 3), 1)))
            [[[0]
              [1]
              [2]]

             [[3]
              [4]]

             [[5]
              [6]
              [7]]]
        """
        return jagged_to_string(self, prefix="[", suffix="]")

    def __repr__(self) -> str:
        """ Return a string for interactive REPL

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3)))
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5, 6, 7]])

            >>> JaggedArray(np.arange(8), (3, 1, (3, 2, 3)))
            JaggedArray([[[0 1 2]],

                         [[3 4]],

                         [[5 6 7]]])

            >>> JaggedArray(np.arange(8), (3, (3, 2, 3), 1))
            JaggedArray([[[0],
                          [1],
                          [2]],

                         [[3],
                          [4]],

                         [[5],
                          [6],
                          [7]]])

            >>> JaggedArray(np.arange(8), (3, (3, 2, 3)), dtype='f4')
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5, 6, 7]], dtype=float32)
        """

        prefix = self.__class__.__name__ + "(["

        if self.dtype in (np.float64, np.int64):
            suffix = "])"
        else:
            suffix = f"], dtype={str(self.dtype)})"

        return jagged_to_string(self, prefix=prefix, suffix=suffix, separator=", ")

    def __getitem__(self, item):
        view = self.data.view()
        cs = np.insert(np.cumsum(self.sizes), 0, 0)
        view = view[cs[item] : cs[item + 1]]
        view.shape = self.shapes[item]
        return view

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
        self._shape = JaggedShape(shape)

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
        return self.shape.to_shapes()

    @shapes.setter
    def shapes(self, shapes: np.ndarray):
        self.shape = JaggedShape.from_shapes(shapes)

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
        return self.shape.sizes

    @property
    def nbytes(self) -> int:
        """ the number of bytes taken up by the jagged array.

        Note:
            This does not factor in shape metadata as of now.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).nbytes
            64

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).nbytes
            144
        """
        return self.data.nbytes

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
        """ the length of the largest subarray along each axis.

        i.e. the shape of the 'convex hull' of the array.
        This would be the shape of the resultant dense array.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).limits
            (3, 3)

            >>> JaggedArray(np.arange(18), (3, (4, 2, 2), (2, 3, 2))).limits
            (3, 4, 3)

        See Also:
            JaggedArray.shape
        """
        return self.shape.limits

    @property
    def jagged_axes(self) -> Tuple[bool]:
        """ The indexes of jagged axes

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).jagged_axes
            (1,)

            >>> JaggedArray(np.arange(16), (3, (3, 2, 3), 2)).jagged_axes
            (1,)

            >>> JaggedArray(np.arange(20), (3, (3, 2, 3), 2, (1, 2, 1))).jagged_axes
            (1, 3)

            >>> JaggedArray(np.arange(8), (3, (3, 2, 3), (1, 1, 1))).jagged_axes
            (1,)
         """
        return self.shape.jagged_axes

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
        # shape is immutable, so this is fine to pass without copy
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
        return JaggedArray(self.data.astype(dtype), self.shape)

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
        from .illife import illife_to_jagged

        return illife_to_jagged(arr)

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
        from .masked import masked_to_jagged

        return masked_to_jagged(arr)

    @classmethod
    def from_array(
        cls, arr: np.ndarray, masked_value: Optional[Any] = np.nan
    ) -> JaggedArray:
        """ Create a jagged array from a (full) array with a masked value.

        Args:
            arr:
                array to convert.

            masked_value:
                The masked value.

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

        if masked_value is np.nan:
            masked = np.ma.masked_array(arr, np.isnan(arr))
        else:
            masked = np.ma.masked_equal(arr, masked_value)
        return cls.from_masked(masked)

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
        from jagged.masked import jagged_to_masked

        return jagged_to_masked(self)

    def to_illife(self, copy=False) -> np.ndarray:
        """ Convert the jagged array to an Illife vector.

        Args:
            copy:
                Whether to return copies or views of the jagged array.

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
        from .illife import jagged_to_illife

        return jagged_to_illife(self, copy=copy)

    def to_array(self, fill_value: Optional[Any] = np.nan) -> np.ndarray:
        """ Convert to a dense array.

        Args:
            fill_value:
                The value to fill in the array.

        Notes:
            Using the default of `np.nan` as the `fill_value` will cause a
            jagged array of integer dtype to be coerced into a float array
            due to the lack of a numpy nan integer type.

        Examples:
            >>> ja = JaggedArray(np.arange(8),  (3, (3, 2, 3))))
            >>> ja.to_array()
            array([[ 0.,  1.,  2.],
                   [ 3.,  4., nan],
                   [ 5.,  6.,  7.]])

            >>> ja.to_array(-1)
            array([[ 0,  1,  2],
                   [ 3,  4, -1],
                   [ 5,  6,  7]])

            >>> JaggedArray(np.arange(33), np.array([[3, 2, 3],
            ...                                      [3, 6, 4]])).to_array(-1)
            array([[[ 0,  1,  2, -1, -1, -1],
                    [ 3,  4,  5, -1, -1, -1],
                    [ 6,  7,  8, -1, -1, -1]],

                   [[ 9, 10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19, 20],
                    [-1, -1, -1, -1, -1, -1]],

                   [[21, 22, 23, 24, -1, -1],
                    [25, 26, 27, 28, -1, -1],
                    [29, 30, 31, 32, -1, -1]]])
        """
        masked = self.to_masked()
        if fill_value is np.nan and np.issubdtype(masked.dtype, np.integer):
            masked = masked.astype(float)

        return masked.filled(fill_value)

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
        return JaggedArray(self.data.clip(a_min=a_min, a_max=a_max), self.shape)

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
        return JaggedArray(self.data.conjugate(), self.shape)

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
            array([0, 1, 2, 3, 4, 5, 6, 7])

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
        from .api import flatten

        return flatten(self)

    def ravel(self) -> np.ndarray:
        """ Ravel the array.

        Creates a **view** of the data.

        Examples:
            >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
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
        from .api import ravel

        return ravel(self)

    @property
    def imag(self) -> JaggedArray:
        """ Get the imaginary part of the jagged array.

        Examples:
            >>> JaggedArray([1j, 1 + 1j, 1 - 1j], (2, (2, 1)))
            JaggedArray([[1, 1],
                         [-1]])
        """
        return JaggedArray(self.data.imag, self.shape)

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
        return JaggedArray(self.data.real, self.shape)

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
            >>> ja.reshape((3, (2, 3, 3)))
            JaggedArray([[0, 1],
                         [2, 3, 4],
                         [5, 6, 7]])
            >>> ja.reshape((3, (3, 4, 3)))
            Traceback (most recent call last):
                ...
            ValueError: cannot reshape jagged array of size 8 into shape (3, (3, 4, 3)) (size 10).
        """
        shape = JaggedShape(shape)
        if shape.size != self.size:
            msg = f"cannot reshape array of size {self.size} into shape {shape} (size {shape.size})"
            raise ValueError(msg)
        return JaggedArray(self.data.copy(), shape)

    def resize(self, shape: ShapeLike) -> JaggedArray:
        """ resize a jagged array in place.

        If resized shape is larger, pad with zeros, otherwise clip the values.

        Args:
            shape:
                the shape of the resized array.

        Examples:
            >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
            >>> ja.resize((2, (3, 2)))
            >>> ja
            JaggedArray([[0, 1, 2],
                         [3, 4]])

            >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
            >>> ja.resize((3, (3, 4, 3)))
            >>> ja
            JaggedArray([[0, 1, 2],
                         [3, 4, 5, 6],
                         [7, 0, 0]])
        """

        shape = JaggedShape(shape)
        self.data.resize(shape.size)
        self.shape = shape

    def squeeze(self, axis: Optional[AxisLike] = None):
        """ Squeeze the given axis.

        This will remove axes from shape with only single dimensional entries.

        Args:
            axis:
                the axes to squeeze.
        >>> jagged.squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3))))
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        Squeezing multiple axes at once:

        >>> jagged.squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3), 1))
        JaggedArray([[0, 1, 2],
                     [3, 4],
                     [5, 6, 7]])

        Squeezing a particular axis:

        >>> jagged.squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3), 1)), axis=-1)
        JaggedArray([[[0, 1, 2]],

                     [[3, 4]],

                     [[5, 6, 7]]])

        >>> _.shape
        (3, 1, (3, 2, 3))

        Squeezing multiple particular axes:

        >>> jagged.squeeze(JaggedArray(np.arange(7), (3, 1, 1, (3, 2, 3), 1)), axis=(1, 2))
        JaggedArray([[[0],
                      [1],
                      [2]],

                     [[3],
                      [4]],

                     [[5],
                      [6],
                      [7]]])

        >>> _.shape
        (3, (3, 2, 3), 1)

        Trying to squeeze an axis with more than one entry:

        >>> jagged.squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3))), axis=2)
        Traceback (most recent call last):
            ...
        ValueError: cannot select an axis to squeeze out which has size not equal to one

        Trying to squeeze the inducing axis:

        >>> jagged.squeeze(JaggedArray(np.arange(7), (3, 1, (3, 2, 3))), axis=0)
        Traceback (most recent call last):
            ...
        ValueError: cannot select an axis to squeeze out which has size not equal to one

        Squeezing the inducing axis when it is only of length one:

        >>> JaggedArray(np.arange(4), (, (1, 2, 2))).squeeze(axis=0)
        array([[0, 1],
               [2, 3]])

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

        Examples:
        >>> import jagged
        >>> ja = JaggedArray(np.arange(8), (3, (3, 2, 3)))
        >>> ja.expand_dims(axis=1)
        JaggedArray([[[0, 1, 2]],

                     [[3, 4]],

                     [[5, 6, 7]]])

        >>> ja.expand_dims(axis=-1)
        JaggedArray([[[0],
                      [1],
                      [2]],

                     [[3],
                      [4]],

                     [[5],
                      [6],
                      [7]]])

        See also:
            :func:`expand_dims`: equivalent standalone function
        """
        from .api import expand_dims

        return expand_dims(self, axis=axis)

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
