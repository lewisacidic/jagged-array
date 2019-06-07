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
from typing import Union

import numpy as np

from .indexing import getitem
from .shape import JaggedShape
from .slicing import canonicalize_index
from .typing import ArrayLike
from .typing import AxisLike
from .typing import DtypeLike
from .typing import JaggedMetadata
from .typing import JaggedShapeLike
from .typing import Number
from .typing import ShapeLike
from .utils import array_to_metadata
from .utils import is_integer
from .utils import jagged_to_string
from .utils import metadata_to_array
from .utils import sanitize_shape


class JaggedArray(np.lib.mixins.NDArrayOperatorsMixin):
    """ Object supporting arrays with jagged axes off an inducing axis.

    Args:
        shape:
            The shape of the data.
        dtype:
            The dtype of the data.
        buffer:
            A :class:`memoryview` of the data.
        offsets:
            A tuple of offsets of the subarrays of the data.
        strides:
            A tuple of the strides of the subarrays of the data.
        order:
            the order of the subarrays of the data.

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

        Using an Iliffe vector:
        >>> JaggedArray.from_iliffe([[0, 1, 2], [3, 4], [5, 6, 7]])
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
        shape: JaggedShapeLike,
        buffer: Optional[ArrayLike] = None,
        strides: Optional[JaggedMetadata] = None,
        dtype: Optional[DtypeLike] = None,
        offsets: Optional[Tuple[int]] = None,
        order: Optional[Union[str, Tuple[str]]] = None,
    ) -> JaggedArray:
        """ Initialize a jagged array.

        Please see `help(JaggedArray)` for more info. """

        self.shape = shape
        self.dtype = dtype
        self.order = order
        self.offsets = offsets
        self.data = buffer
        self.strides = strides

    @property
    def data(self) -> memoryview:
        """ 1D array storing all entries of the  array.

        Examples:
            >>> ja = jagged.arange(shape=(3, 2, 3))
            >>> ja.data
            <memory at ...>

            >>> list(ja.data)
            [0, 1, 2, 3, 4, 5, 6, 7]
        """
        return self._data

    @data.setter
    def data(self, value: ArrayLike):
        if value is None:
            value = np.empty(self.size).data
        if not isinstance(value, memoryview):
            value = np.asarray(value).data
        self._data = value

    @property
    def shape(self) -> JaggedShape:
        """ Tuple of array dimensions.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).shape
            (3, (3, 2, 3))

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).shape
            (3, (4, 2, 2), (2, 3, 2))

        See Also:
            JaggedArray.shapes
        """
        return self._shape

    @shape.setter
    def shape(self, shape: JaggedShapeLike):
        self._shape = sanitize_shape(shape)

    @property
    def shape_array(self) -> np.ndarray:
        """ the shapes of subarrays as an array

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).shape
            array([[3],
                   [2],
                   [3]])

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).shape
            array([[4, 2],
                   [2, 3],
                   [2, 2]])
        """
        return metadata_to_array(self.shape[0], self.shape[1:])

    @property
    def dtype(self) -> np.dtype:
        """ the dtype of the contained data.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).dtype
            dtype('int64')

            >>> jagged.arange(shape=(3, (3, 2, 3)), dtype="f4").dtype
            dtype('float32')
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value: DtypeLike):
        # automatically picks the default dtype if `None` is passed
        self._dtype = np.dtype(value)

    @property
    def offsets(self) -> tuple:
        """ the offsets of the subarrays along the inducing axis.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).offsets
            (0, 24, 40, 64)

            >>> jagged.arange(shape=(3, 2, (3, 2, 3))).offsets
            (0, 48, 80, 128)
        """
        return self.default_offsets if self._offsets is None else self._offsets

    @offsets.setter
    def offsets(self, value):
        self._offsets = value

    @property
    def default_offsets(self):
        """ the offsets for the given shape

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3)), dtype="i8").default_offsets
            (0, 24, 40)

            >>> jagged.arange(shape=(3, (3, 2, 3)), dtype="i4").default_offsets
            (0, 12, 20)

            >>> jagged.arange(shape=(3, 2, (3, 2, 3)), dtype="i8").default_offsets
            (0, 48, 80)

            >>> jagged.arange(shape=(3, (3, 2, 3), 2)), dtype="i8").default_offsets
            (0, 48, 80)

            >>> jagged.arange(shape=(3, 2, (3, 2, 3)), dtype="i4").default_offsets
            (0, 24, 40)

            >>> jagged.arange(shape=(3, (3, 2, 3)), dtype="b1").default_offsets
            (0, 3, 5)
        """
        return (0,) + tuple(np.cumsum(self.sizes) * self.dtype.itemsize)[:-1]

    @property
    def offsets_array(self):
        """ the offsets of the subarrays as an array

        Examples:
           >>> jagged.arange(shape=(3, (3, 2, 3))).offsets
            array([0, 24, 40])

            >>> jagged.arange(shape=(3, 2, (3, 2, 3))).offsets
            array([0, 48, 80])
        """
        return np.asarray(self.offsets)

    @property
    def strides(self):
        """ the strides of the subarrays along the inducing axis.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).offsets
            (8,)

            >>> jagged.arange(shape=(3, 2, (3, 2, 3))).offsets
            ((24, 16, 24), 8)
        """
        return self._strides

    @strides.setter
    def strides(self, value: JaggedMetadata):
        self._strides = self.default_strides if value is None else value

    @property
    def default_strides(self):
        """ the strides for the given shape.

        Examples:
            >>> JaggedArray((3, (3, 2, 3)), dtype="i8").default_strides
            (8,)

            >>> JaggedArray((3, (3, 2, 3)), dtype="i4").default_strides
            (8,)

            >>> JaggedArray((3, 2, (3, 2, 3)), dtype="i8").default_strides
            ((24, 16, 24), 8)

            >>> JaggedArray((3, 2, (3, 2, 3))), dtype="i8", order="F").default_strides
            (8, 16)

            >>> JaggedArray((3, 2, (3, 2, 3)), dtype="i4").default_strides
            ((12, 8, 12), 8)

            >>> JaggedArray((3, (3, 2, 3), 2)), dtype="i8").default_strides
            (16, 8)

            >>> JaggedArray((3, (3, 2, 3)), dtype="b1").default_strides
            (1,)
        """
        if self.order != "C":
            raise NotImplementedError(
                "calculating strides is only implemented for C order subarrays"
            )

        itemsize = self.dtype.itemsize

        cp = np.cumprod(self.shape_array[:, ::-1], axis=1)
        cp = cp[:, -2::-1]
        return array_to_metadata(
            itemsize * np.hstack([cp, np.ones((len(cp), 1), cp.dtype)])
        )

    @property
    def strides_array(self) -> np.ndarray:
        """ the strides of subarrays as an array

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).strides_array
            array([[3],
                   [2],
                   [3]])

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).strides_array
            array([[4, 2],
                   [2, 3],
                   [2, 2]])
        """
        return metadata_to_array(len(self), self.strides)

    @property
    def order(self):
        """ The memory order of the subarrays. """

        return self._order

    @order.setter
    def order(self, value):
        if value is None:
            value = "C"
        elif value not in ("C", "F"):
            raise ValueError(f"{value} is not a valid order.")

        self._order = value

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
            >>> jagged.arange(shape=(3, (3, 2, 3)))
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5, 6, 7]])

            >>> jagged.arange(shape=(3, 1, (3, 2, 3)))
            JaggedArray([[[0 1 2]],

                         [[3 4]],

                         [[5 6 7]]])

            >>> jagged.arange(shape=(3, (3, 2, 3), 1))
            JaggedArray([[[0],
                          [1],
                          [2]],

                         [[3],
                          [4]],

                         [[5],
                          [6],
                          [7]]])

            >>> jagged.arange(shape=(3, (3, 2, 3)), dtype="f4")
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
        index = canonicalize_index(item, self.shape)
        return getitem(self, index)

    @property
    def size(self) -> int:
        """ the number of elements in the jagged array.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).size
            8

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).size
            18
        """
        return self.sizes.sum()

    @property
    def sizes(self) -> Tuple[int]:
        """ the sizes of the subarrays along the inducing axis.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).sizes
            (3, 2, 3)

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).sizes
            (8, 6, 4)
        """
        return tuple(self.shape_array.prod(axis=1))

    @property
    def nbytes(self) -> int:
        """ the number of bytes taken up by the jagged array.

        Note:
            This does not factor in shape metadata as of now.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).nbytes
            64

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).nbytes
            144
        """
        return self.data.nbytes

    @property
    def ndim(self) -> int:
        """ the number of dimensions.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).ndim
            2

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).ndim
            3
        """
        return len(self.shape)

    @property
    def minshape(self) -> np.ndarray:
        """ the length of the smallest subarray along each axis.

        i.e. the shape of the 'smooth core' of the array.
        This would be the shape of the resultant smooth array.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).minshape
            (3, 3)

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).minshape
            (3, 4, 3)

        See Also:
            JaggedArray.shape
            JaggedArray.maxshape
        """
        return tuple(ax if is_integer(ax) else min(ax) for ax in self.shape)

    @property
    def maxshape(self) -> tuple:
        """ the length of the largest subarray along each axis.

        i.e. the shape of the 'convex hull' of the array.
        This would be the shape of the resultant masked array.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).maxshape
            (3, 3)

            >>> jagged.arange(shape=(3, (4, 2, 2), (2, 3, 2))).maxshape
            (3, 4, 3)

        See Also:
            JaggedArray.shape
            JaggedArray.minshape
        """
        return tuple(ax if is_integer(ax) else max(ax) for ax in self.shape)

    @property
    def jagged_axes(self) -> Tuple[int]:
        """ The indexes of jagged axes

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).jagged_axes
            (1,)

            >>> jagged.arange(shape=(3, (3, 2, 3), 2)).jagged_axes
            (1,)

            >>> jagged.arange(shape=(3, (3, 2, 3), 2, (1, 2, 1))).jagged_axes
            (1, 3)

            >>> jagged.arange(shape=(3, (3, 2, 3), (1, 1, 1))).jagged_axes
            (1,)

        See Also:
            JaggedArray.is_jagged
        """
        return tuple(i for i, ax in enumerate(self.shape) if isinstance(ax, tuple))

    @property
    def is_jagged(self) -> bool:
        """ whether the array is jagged.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3)))
            True

            >>> jagged.arange(shape=(3, 3))
            False

        See also:
            JaggedArray.jagged_axes
        """

        return any(self.jagged_axes)

    def copy(self) -> JaggedArray:
        """ copy the jagged array.

        Examples:
            >>> ja = jagged.arange(shape=(3, (3, 2, 3)))
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
        # everything but data is immutable, so this is fine to pass without copy
        return JaggedArray(
            self.shape,
            buffer=self.data.copy(),
            strides=self.strides,
            dtype=self.dtype,
            offsets=self.offsets,
            order=self.order,
        )

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
        res.dtype = dtype
        return res

    @classmethod
    def from_iliffe(cls, arr: ArrayLike, dtype: DtypeLike = None) -> JaggedArray:
        """ Create a jagged array from an Iliffe vector (array of arrays).

        Args:
            arr:
                Iliffe vector to convert.
            dtype:
                The dtype of the resulting jagged array.

        Examples:
            >>> JaggedArray.from_iliffe([[0, 1, 2],
            ...                          [3, 4],
            ...                          [5, 6, 7]])
            JaggedArray([[0, 1, 2],
                         [3, 4],
                         [5, 6, 7]])

            >>> JaggedArray.from_iliffe([[[0, 1, 2],
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
        from .iliffe import iliffe_to_jagged

        return iliffe_to_jagged(arr, dtype)

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

    def to_iliffe(self, copy=False) -> np.ndarray:
        """ Convert the jagged array to an Illife vector.

        Args:
            copy:
                Whether to return copies or views of the jagged array.

        Examples:
            >>> JaggedArray(np.arange(8), (3, (3, 2, 3))).to_iliffe()
            array([array([0, 1, 2]),
                   array([3, 4]),
                   array([5, 6, 7])], dtype=object)

            >>> JaggedArray(np.arange(33), (3, (3, 2, 3), (3, 6, 4))).to_iliffe()
            array([array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8]]),
                   array([[ 9, 10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19, 20]]),
                   array([[21, 22, 23, 24],
                          [25, 26, 27, 28],
                          [29, 30, 31, 32]])], dtype=object)
        """
        from .iliffe import jagged_to_iliffe

        return jagged_to_iliffe(self, copy=copy)

    def to_array(self, copy=True):
        """ return a numpy array of the JaggedArray given that it is smooth.

        Examples:
            >>> jagged.arange(shape=(2, 2)).to_array()
            array([[0, 1],
                   [2, 3]])

            >>> jagged.arange(shape=(2, (3, 2))).to_array()
            Traceback (most recent call last):
                ...
            ValueError: Cannot create a smoothe array for jagged. Try `to_iliffe` or `to_masked`.
        """
        if self.is_jagged:
            msg = "Cannot create a smoothe array from jagged. Try `to_iliffe` or `to_masked`."
            raise ValueError(msg)
        else:
            return np.array(self, copy=copy)

    def smoothe(self, axis=None):
        """ smoothe a jagged axis by removing jagged ends.

        Args:
            jarr:
                the jagged axis.
            axis:
                the axis to smoothe.  When passed `None`, smoothe all axes and
                return a numpy array.

        Examples:
            >>> jagged.arange(shape=(3, (3, 2, 3))).smoothe()
            array([[0, 1],
                [3, 4],
                [5, 6]])

            >>> jagged.arange(shape=(3, (3, 2, 3))).smoothe(axis=1)
            JaggedArray([[0, 1],
                        [3, 4],
                        [5, 6]])

            >>> jagged.arange(shape=(3, (3, 2, 3), (2, 3, 2))).smoothe(axis=(1, 2))
            JaggedArray([[[ 0,  1],
                        [ 2,  3]],

                        [[ 6,  7],
                        [ 9, 10]],

                        [[12, 13],
                        [14, 15]]])

            >>> jagged.arange(shape=(3, (3, 2, 3))).smoothe(axis=0)
            Traceback (most recent call last):
                ...
            ValueError: axis 0 is not jagged and so cannot be smoothed.
        """
        from .api import smoothe

        return smoothe(self, axis=axis)

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
        return np.asarray(self.data).flat

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
        return JaggedArray(
            self.shape,
            buffer=np.asarray(self.data).imag,
            dtype=self.dtype,
            offsets=self.offsets,
            strides=self.strides,
            order=self.order,
        )

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
        return JaggedArray(
            self.shape,
            buffer=np.asarray(self.data).real,
            dtype=self.dtype,
            offsets=self.offsets,
            strides=self.strides,
            order=self.order,
        )

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

        return squeeze(self, axis=axis)

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
