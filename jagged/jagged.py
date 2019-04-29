#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

""" Jagged array support for scikit-chem. """

from collections.abc import Iterable
from functools import partial

import numpy as np


def infer_nan(dtype):
    """ Infer the nan value for a given dtype

    Examples:
        >>> infer_nan(np.int32)
        nan

        >>> infer_nan(np.float64)
        nan

        >>> infer_nan(np.dtype('S4'))
        'N/A'

        >>> infer_nan(np.object_))
        None
    """

    if np.issubdtype(dtype, np.number):
        return np.nan
    elif np.issubdtype(dtype, np.str):
        return "N/A"
    else:
        return None


def is_float(obj):
    """ Whether an object is a float. """

    return isinstance(obj, (float, np.float))


class JaggedArray(object):
    """ Object supporting arrays with jagged shapes off the zero'th axis. """

    def __init__(self, data, shape):
        """ Initialize a jagged array.

        Args:
            data (np.ndarray):
                The data as a one dimensional array.
            shape (np.ndarray):
                The shape of the data along the zero'th axis.
        """

        self._data = self.__cumsum = self._shape = None
        self._data = np.array(data)
        self._shape = np.array(shape)
        self._verify_integrity()

    @property
    def data(self):
        """ data (np.ndarray): 1D array storing all entries of the  array. """
        return self._data

    @data.setter
    def data(self, val):
        old_data = self.data
        self._data = val
        try:
            self._verify_integrity()
        except ValueError:
            self._data = old_data
            raise

    @property
    def shape(self):
        """np.ndarray: the shapes of subarrays along the zero'th axis.

        dims: (D, m)
        where D is the number of dimensions and m is the length along the
        zero'th axis. """

        return self._shape

    @shape.setter
    def shape(self, val):
        val = np.asarray(val)
        old_shape = self.shape
        self._shape = val
        try:
            self._verify_integrity()
        except ValueError:
            self._shape = old_shape
            raise

    @property
    def sizes(self):
        """ np.ndarray: the sizes of the subarrays along the zero'th axis.

        dims: (m)
        where m is the length along along the zero'th axis.
        """

        return self.shape.prod(axis=0)

    @property
    def size(self):
        """ int: the number of elements in the jagged array. """

        return self.sizes.sum()

    @property
    def ndim(self):
        """ int: the number of dims. """

        return 1 + len(self.shape)

    @property
    def dtype(self):
        """ the dtype of the contained data. """

        return self.data.dtype

    @property
    def limits(self):
        """ np.ndarray: the shape of the largest array for each dimension. """

        return np.insert(self.shape.max(axis=1), 0, self.shape.shape[1])

    @property
    def _cumsum(self):
        """ np.ndarray: indices into the data along the zero'th axis. """

        if not hasattr(self, "__cumsum"):
            self.__cumsum = np.insert(np.cumsum(self.sizes), 0, 0)
        return self.__cumsum

    @classmethod
    def from_aoa(cls, arr):
        """ Create a jagged array from a numpy array of arrays.

        Args:
            arr (np.ndarray):
                Numpy array of arrays to convert.

        Returns:
            JaggedArray

        Examples:
            >>> arr = np.array([np.array([0, 1, 2]),
            ...                 np.array([3, 4]),
            ...                 np.array([5, 6, 7])])
            >>> JaggedArray.from_aoa(arr)
            JaggedArray(data=[0 1 2 3 4 5 6 7],
                        shape=[[3 2 3]],
                        dtype=int64)
        """

        return cls(
            np.concatenate([sub.flatten() for sub in arr]),
            np.array([sub.shape for sub in arr]).T,
        )

    @classmethod
    def from_masked(cls, arr):
        """ Create a jagged array from a masked numpy array.

        Args:
            arr (np.ma.masked_array):
                Masked numpy array to convert.

        Returns:
            JaggedArray

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
            JaggedArray(data=[0 1 2 3 4 5 6 7 8],
                        shape=[[3 2 1 3]],
                        dtype=int64)

        Notes:
            The first masked value in a given direction is assumed to be the
            end of the array.
        """
        mask = arr.mask
        return cls._from_arr_and_mask(arr.compressed(), mask)

    @classmethod
    def _from_arr_and_mask(cls, arr, mask):
        def get_shape(mask, axis=1):
            res = (~mask).argmin(axis=axis)
            res = res.max(axis=-1) if res.ndim > 2 else res
            res[res == 0] = mask.shape[axis]
            return res

        shapes = np.vstack([get_shape(mask, axis=i) for i in range(1, len(mask.shape))])
        return cls(arr, shapes)

    @classmethod
    def from_array(cls, arr, masked_value=None):
        """ Create a jagged array from a (full) array with a masked value.

        Args:
            arr (np.ndarray):
                array to convert.

            masked_value (any):
                The masked value.  If no value is passed and the array is
                compatible with float, this will be `nan`, otherwise `None`.

        Returns:
            JaggedArray

        Examples:
            >>> arr = np.array([[     0.,     1.,     2.],
            ...                 [     3.,     4., np.nan],
            ...                 [     5., np.nan, np.nan],
            ...                 [     6.,     7.,     8.]])
            >>> JaggedArray.from_array(arr).astype(np.int64)
            JaggedArray(data=[0 1 2 3 4 5 6 7 8],
                        shape=[[3 2 1 3]],
                        dtype=int64)
        """
        if masked_value is None:
            masked_value = infer_nan(arr.dtype)

        if masked_value == np.nan:
            mask = np.isnan(arr)
        else:
            mask = np.equal(arr, masked_value)
        return cls._from_arr_and_mask(arr[~mask], mask)

    @classmethod
    def from_format(cls, arr, format, **kwargs):
        """ Instantiate a JaggedArray from a jagged format.

        Args:
            arr (np.ndarray):
                array to convert.

        Keyword Args:
            are passed onto the initializer.

        Returns:
            JaggedArray
        """

        try:
            return getattr(cls, "from_" + format)(arr, **kwargs)
        except AttributeError:
            raise ValueError("{} is not a valid jagged format.".format(format))

    def copy(self):
        """ copy the jagged array. """

        return self.__class__(self.data.copy(), self.shape.copy())

    def astype(self, dtype, copy=True):
        """ the array with the data as a given data type.

        Args:
            dtype (np.dtype):
                the numpy dtype to use to represent data.
            copy (bool):
                whether to copy the data, or make the change in place.

        Returns:
            JaggedArray
        """
        res = self.copy() if copy else self
        res.data = self.data.astype(dtype)
        return res

    def _verify_integrity(self):
        """ Verify that the jagged array is acceptable.

        This checks that:
            - the data is 1D
            - the shape is 2D
            - the number of entries match the sizes of the array.

        Returns:
            bool
        """
        if len(self.data.shape) != 1:
            raise ValueError(
                "Data array must be one dimensional "
                "(is {})".format(len(self.data.shape))
            )

        if len(self.shape.shape) != 2:
            raise ValueError(
                "Shape array must be two dimensional "
                "(is {})".format(len(self.shape.shape))
            )

        shape_size, data_size = self._cumsum[-1], self.data.size

        if not shape_size == data_size:
            raise ValueError(
                "Size of data ({data_size}) does not match that "
                "of the given shapes ({shape_size}).".format(
                    data_size=data_size, shape_size=shape_size
                )
            )

    def _mask(self):
        """ the mask for a dense array for the given shapes. """
        mask = np.ones(self.limits, dtype=bool)
        for ax, shape, limit in zip(
            range(1, len(self.limits)), self.shape, self.limits[1:]
        ):
            ax_mask = np.arange(limit) < np.expand_dims(shape, 1)
            new_shape = np.ones(len(self.limits), dtype=int)
            new_shape[0], new_shape[ax] = self.limits[0], limit
            mask = mask & ax_mask.reshape(*new_shape)
        return mask

    def to_masked(self):
        """ convert the array to a dense masked array.

        Returns:
            np.ma.MaskedArray

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
        mask = self._mask()
        res = np.ma.masked_all(self.limits, dtype=self.dtype)
        res[mask] = self.data
        return res

    def to_aoa(self):
        """ Return a numpy array of arrays.

        Returns:
            np.ndarray

        Examples:
            >>> JaggedArray(np.arange(8), np.array([[3, 2, 3]])).to_aoa()
            array([array([0, 1, 2]),
                   array([3, 4]),
                   array([5, 6, 7])], dtype=object)

            >>> JaggedArray(np.arange(33), np.array([[3, 2, 3],
            ...                                      [3, 6, 4]])).to_aoa()
            array([array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8]]),
                   array([[ 9, 10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19, 20]]),
                   array([[21, 22, 23, 24],
                          [25, 26, 27, 28],
                          [29, 30, 31, 32]])], dtype=object)
        """
        arr = np.array_split(self.data, self._cumsum[1:])
        return np.array([res.reshape(*shape) for res, shape in zip(arr, self.shape.T)])

    def to_array(self, fill_value=None):
        """ Convert to a dense array.

        Args:
            fill_value (any):
                The value to fill in the array.  If `None` (as default) and the
                array can be converted to floats, we will use `np.nan`.

        Returns:
            np.ndarray

        Examples:
            >>> JaggedArray(np.arange(8), np.array([[3, 2, 3]])).to_array()
            array([array([  0.,   1.,   2.]),
                   array([  3.,   4.,  nan]),
                   array([  5.,   6.,   7.])], dtype=np.float64)
        """
        if fill_value is None:
            fill_value = infer_nan(self.dtype)

        tmp = self.astype(float) if is_float(fill_value) else self
        return tmp.to_masked().filled(fill_value=fill_value)

    def to_format(self, format, **kwargs):
        """ Convert the jagged array to a different format.

        This is a convenience function around `to_masked`, `to_aoa`, etc.

        Args:
            format (str):
                The type of array.

        Returns:
            JaggedArray | np.array
        """

        try:
            return getattr(self, "to_" + format)(**kwargs)
        except AttributeError:
            raise ValueError("{} is not a valid jagged format.".format(format))

    asformat = to_format  # consistency with scipy.sparse

    def __eq__(self, other):
        """ Whether one JaggedArray equals another. """

        return np.array_equal(self.data, other.data) and np.array_equal(
            self.shape, other.shape
        )

    @classmethod
    def from_broadcast(cls, arr, shape):
        return cls(np.repeat(arr, shape.prod(axis=0)), shape)

    def __array__(self, *args, **kwargs):
        """ Numpy array interface for ufuncs.

        This just gives ufuncs the data to operate on. """

        return self.data

    def __array_wrap__(self, result, **kwargs):
        """ Numpy array interface for ufuncs.

        This takes the result of a ufunc and rejaggedizes it. """

        return self.__class__(result, self.shape)

    def _unitary_op(self, op):
        return self.__class__(op(self.data), self.shape)

    def __neg__(self):

        return self._unitary_op(np.negative)

    def _binary_elementwise_op(self, other, op):
        if isinstance(other, JaggedArray):
            if not np.array_equal(other.shape, self.shape):
                raise ValueError(
                    "operands cound not be broadcast "
                    "together with shapes {} and "
                    "{}.".format(self.shape, other.shape)
                )
            return self.__class__(op(self.data, other.data), self.shape)
        else:
            other = np.asanyarray(other)
            if other.ndim == 2:
                # try to broadcast
                if other.shape[0] != len(self):
                    raise ValueError(
                        "operands could not be broadcast "
                        "together with zero-axis lengths {} and "
                        "{}.".format(len(self), other.shape[0])
                    )
                return self._binary_elementwise_op(
                    self.from_broadcast(other, self.shape), op
                )
            elif other.ndim > 2:
                raise ValueError(
                    "Could not broadcast dense array of shape {} "
                    "to jagged array.".format(other.shape)
                )
            # otherwise we have single
            return self.__class__(op(self.data, other), self.shape)

    def __add__(self, other):
        """ Add Jagged by a value. """

        return self._binary_elementwise_op(other, np.add)

    def __mul__(self, other):
        """ Multiply JaggedArray by a value. """

        return self._binary_elementwise_op(other, np.multiply)

    def __truediv__(self, other):
        """ True divide a JaggedArray by a value. """

        return self._binary_elementwise_op(other, np.true_divide)

    def __floordiv__(self, other):

        return self._binary_elementwise_op(other, np.floor_divide)

    def __sub__(self, other):

        return self._binary_elementwise_op(other, np.subtract)

    def __pow__(self, power, modulo=None):
        if modulo:
            raise NotImplementedError("modulo argument not implemented.")
        return self._binary_elementwise_op(power, np.power)

    def __mod__(self, other):
        return self._binary_elementwise_op(other, np.mod)

    def clip(self, a_min=None, a_max=None):
        """ Clip the values of the array.

        Args:
            a_min (float or None):
                Lower bound of clipping interval.  Values below this will be
                set to this value.
            a_max (float or None):
                Upper bound of clipping interval.  Values above this will be
                set to this value.

        Returns:
            JaggedArray

        Examples:
            >>> JaggedArray(np.arange(7), [2, 3, 2]).clip(2, 5)
            JaggedArray(data=[ 2 2 2 3 4 5 5],
                        shape=[[2, 3, 2]],
                        dtype=int64)
        """

        return self._unitary_op(partial(np.clip, a_min=a_min, a_max=a_max))

    def conjugate(self):
        """ Return the element-wise complex conjugate.

        The complex conjugate of a number is obtained by changing the sign of
        its imaginary part.

        Returns:
            JaggedArray

        Examples:
            >>> JaggedArray([np.complex(0, 1), np.complex(1, 1), np.complex(1, -1)], [[2, 1]]).conjugate()
            JaggedArray(data=[0.-1.j 1.-1.j 1.+j],
                        shape=[[2 1]],
                        dtype=complex128)
        """
        return self._unitary_op(np.conjugate)

    conj = conjugate

    def fill(self, value):
        """ Fill the array with a scalar value.

        Args:
            value (any):
                All elements of `a` will be assigned this value.

        Examples:
            >>> ja = JaggedArray(np.arange(7), [[3, 2, 3]])
            >>> ja.fill(0)
            >>> ja
            JaggedArray(data=[0 0 0 0 0 0 0],
                        shape=[[3 2 3]],
                        dtype=int64)

        Returns:
            None
        """

        self.data[...] = value

    @property
    def flat(self):
        return self.data.flat

    def flatten(self):
        """ Flatten the array.

        Returns:
            np.ndarray
        """

        return self.data

    @property
    def imag(self):
        return self._unitary_op(np.imag)

    @imag.setter
    def imag(self, values):
        # TODO: broadcasting
        self.data.imag = values

    @property
    def real(self):
        return self._unitary_op(np.real)

    @real.setter
    def real(self, values):
        # TODO: broadcasting
        self.data.real = values

    def _reduce_op(self, op, axis=None, **kwargs):
        # easy if axis is none
        if axis is None:
            return op(self.data)

        aoa = self.to_aoa()

        axis = np.sort(np.atleast_1d(axis))
        # then it is applying to all. Will get a dense array.
        if np.array_equal(axis, np.arange(1, self.ndim)):
            return np.asarray([op(arr) for arr in aoa])

        if axis[0] == 0:
            raise ValueError(
                "reduce operations through the zero'th axis are " "not defined."
            )

        return JaggedArray.from_aoa(
            np.array([op(arr, axis=tuple(axis - 1)) for arr in aoa])
        )

    def all(self, **kwargs):
        return self._reduce_op(np.all, **kwargs)

    def any(self, **kwargs):
        return self._reduce_op(np.any, **kwargs)

    def argmax(self, **kwargs):
        return self._reduce_op(np.argmax, **kwargs)

    def argmin(self, **kwargs):
        return self._reduce_op(np.argmin, **kwargs)

    def cumprod(self, **kwargs):
        return self._reduce_op(np.cumprod, **kwargs)

    def cumsum(self, **kwargs):
        return self._reduce_op(np.cumsum, **kwargs)

    def max(self, **kwargs):
        return self._reduce_op(np.max, **kwargs)

    def mean(self, **kwargs):
        return self._reduce_op(np.mean, **kwargs)

    def min(self, **kwargs):
        return self._reduce_op(np.min, **kwargs)

    def sum(self, **kwargs):
        return self._reduce_op(np.sum, **kwargs)

    def prod(self, **kwargs):
        return self._reduce_op(np.prod, **kwargs)

    def ptp(self, **kwargs):
        return self._reduce_op(np.ptp, **kwargs)

    def put(self, indices, values):
        raise NotImplementedError("put for a jagged array is not supported.")

    ravel = flatten

    def repeat(self, n):
        raise NotImplementedError("repeat for a jagged array is not " "supported.")

    def reshape(self, shape):
        """ reshape the arrays. """

        shape = np.asarray(shape)

        if not np.array_equal(np.prod(shape, axis=0), np.prod(self.shape, axis=0)):
            raise ValueError("total size of new array must be unchanged.")
        else:
            data = np.concatenate(
                [sub.reshape(shp).flatten() for sub, shp in zip(self.to_aoa(), shape.T)]
            )
            new = self[...]
            new.data = data
            new.shape = shape
        return new

    def resize(self, shape):
        """ resize the arrays. """

        new = self[...]
        size = np.prod(shape, axis=1).sum()
        new.data.resize(size)
        new.shape = shape
        return new

    def round(self, *args, decimals=0):
        decimals = args[0] if len(args) else decimals
        return self._unitary_op(partial(np.round, decimals=decimals))

    def std(self, **kwargs):
        return self._reduce_op(np.std, **kwargs)

    def sort(self):
        raise NotImplementedError("sorting a jagged array is not supported.")

    def take(self, arr):
        raise NotImplementedError("take is not supported.")

    def var(self, **kwargs):
        return self._reduce_op(np.var, **kwargs)

    def nonzero(self):
        top = np.repeat(np.arange(len(self)), self.sizes)[self.data.nonzero()]
        bot = np.hstack([np.vstack(arr.nonzero()) for arr in self.to_aoa()])
        return (top,) + tuple(bot)

    def squeeze(self, axis=None):
        """ Squeeze the given axis.

        This will remove axes from shape with only single dimensional entries.

        Args:
            axis (Iterable | int | None):
                the axes to squeeze.

        Returns:
            JaggedArray:
                The array with the axes removed.  This does not copy the data.
        """

        if axis is None:
            axis = range(self.shape.shape[0])
        elif axis == -1:
            axis = self.shape.shape[0]
        if not isinstance(axis, Iterable):
            axis = [axis]
        axis = [ax for ax in axis if (self.shape[ax] == 1).all(axis=0)]
        self.shape = np.delete(self.shape, axis, axis=0)
        return self

    def expand_dims(self, axis=-1):
        """ Expand dimensions.

        See full documenation for :func:`expand_dims`. """

        return expand_dims(self, axis=axis)

    def digitize(self, bins, right=False):
        """ Return the indices of the bins for each value in array.

        Args:
            bins (array_like):
                Array of 1-dimensional, monotonic bins.

            right (bool):
                Whether the intervals include the right or the left bin edge.
        """
        return self.__class__(np.digitize(self.data, bins, right=right), self.shape)

    def trace(self):
        raise NotImplementedError("trace of a jagged array is not implemented")

    def __getitem__(self, item):
        """ Slice into the zero'th axis. """

        if item == Ellipsis:
            return JaggedArray(data=self.data[...], shape=self.shape[...])
        elif isinstance(item, slice):
            # slow but works
            return self.__class__.from_aoa(self.to_aoa()[item])
        else:
            return self.data[slice(*self._cumsum[item : item + 2])].reshape(
                self.shape[:, item]
            )

    def __len__(self):
        """ The length along the zero'th axis. """

        return self.shape.shape[1]

    def __repr__(self):
        """ A string representation of the array. """

        thresh = np.get_printoptions()["threshold"]
        np.set_printoptions(threshold=20)
        extra_chars = len(self.__class__.__name__)
        arr_str = "data=" + str(self.data).replace("\n", "\n" + " " * (extra_chars + 6))
        shape_str = (
            " " * extra_chars
            + " shape="
            + str(self.shape).replace("\n", "\n" + " " * (extra_chars + 7))
        )
        dtype_str = " " * extra_chars + " dtype=" + str(self.dtype)
        np.set_printoptions(threshold=thresh)
        return "{klass}({data},\n{shape},\n{dtype})".format(
            klass=self.__class__.__name__,
            data=arr_str,
            shape=shape_str,
            dtype=dtype_str,
        )

    def save(self, filepath):
        with open(filepath, "wb") as f:
            np.savez_compressed(f, data=self.data, shape=self.shape)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return cls(**np.load(f))


def expand_dims(arr, axis=-1):
    """ Add a dimension.

    Args:
        axis (int):
            The axis after which to add the dimension.

    Examples:
        >>> jarr = JaggedArray(np.arange(8), [[3, 2, 3]]); jarr
        JaggedArray(data=[0 1 2 3 4 5 6 7],
                    shape=[[3 2 3]],
                    dtype=int64)

        >>> expand_dims(jarr, axis=1)
        JaggedArray(data=[0 1 2 3 4 5 6 7],
                    shape=[[3 2 3],
                           [1 1 1]],
                    dtype=int64)

        >>> expand_dims(jarr, -1)
        JaggedArray(data=[0 1 2 3 4 5 6 7],
                    shape=[[3 2 3],
                           [1 1 1]],
                    dtype=int64)
    """

    if axis < 0:
        axis += arr.ndim

    return arr.reshape(np.insert(arr.shape, axis, np.ones(len(arr), int), axis=0))


def collapse_dims(arr, axis=-1, inplace=False):
    """ Collapse a dimension to the lower one.

    Examples:
        >>> ja = JaggedArray(np.arange(33), np.array([[3, 2, 3],
        ...                                           [3, 6, 4]]))

        >>> collapse_dims(ja, axis=2)
        JaggedArray(data=[ 0  1  2 ..., 30 31 32],
        shape=[[9 12 12]],
        dtype=int64)
    """

    assert axis != 0, "cannot collapse the zero'th axis"

    if axis < 0:
        axis += arr.ndim

    keepdims = np.delete(np.arange(arr.ndim), (axis - 1, axis - 2))
    newshape = arr.shape[axis - 2] * arr.shape[axis - 1]

    if not keepdims.size:
        shape = np.expand_dims(newshape, 0)
    else:
        shape = np.concatenate([arr.shape[: axis - 2], newshape], axis=1)

    if not inplace:
        arr = arr.copy()
    arr.shape = shape
    return arr


def concatenate(objs, axis=0):
    """ Concatenate data along axes for jagged arrays.

    Args:
        objs (iterable[JaggedArray]):
            The jagged arrays to concatenate.

        axis (int):
            The axis along which to concatenate.

    Returns:
        JaggedArray

    Examples:
        >>> ja = JaggedArray(np.arange(33), np.array([[3, 2, 3],
        ...                                           [3, 6, 4]]))

        >>> concatenate([ja, ja], axis=0)
        JaggedArray(data=[ 0  1  2 ..., 30 31 32],
            shape=[[3 2 3 3 2 3]
                   [3 6 4 3 6 4]],
            dtype=int64)

        >>> concatenate([ja, ja], axis=1)
        JaggedArray(data=[ 0  1  2 ..., 30 31 32],
                    shape=[[6 4 6]
                           [3 6 4]],
                    dtype=int64)

        >>> concatenate([ja, ja], axis=2)
        JaggedArray(data=[ 0  1  2 ..., 30 31 32],
            shape=[[ 3  2  3]
                   [ 6 12  8]],
            dtype=int64)
    """

    assert all(
        isinstance(obj, JaggedArray) for obj in objs
    ), "all operands must be `JaggedArray`s"
    assert all(
        obj.ndim == objs[0].ndim for obj in objs[1:]
    ), "all operands must be of the same dimensions"

    if len(objs) == 1:
        return objs[0]

    if axis == 0:
        data = np.concatenate([obj.data for obj in objs])
        shape = np.concatenate([obj.shape for obj in objs], axis=1)
    else:
        if axis < 0:
            axis += objs[0].ndim
        shapes = np.dstack([obj.shape for obj in objs])
        keepaxes = np.delete(np.arange(objs[0].ndim - 1), axis - 1, axis=0)
        first_shape = shapes[keepaxes, ..., :1]
        assert (
            first_shape == shapes[keepaxes, ..., 1:]
        ).all(), "all shapes other than the concatenation axis must be equal"
        shape = shapes[axis - 1].sum(axis=1)
        shape = np.insert(shapes[keepaxes, :, 0], axis - 1, shape, axis=0)
        data = [obj.to_aoa() for obj in objs]
        data = np.hstack(
            [np.concatenate(subs, axis=axis - 1).flatten() for subs in zip(*data)]
        )
    return JaggedArray(data=data, shape=shape)


def stack(objs):
    """ Stack JaggedArrays on a new axis.

    Args:
        objs (iterable[JaggedArray]):
            The jagged arrays to stack.

    Returns:
        JaggedArray

    Examples:
        >>> ja = JaggedArray(np.arange(33), np.array([[3, 2, 3],
        ...                                           [3, 6, 4]]))

        >>> stack([ja, ja])
        JaggedArray(data=[ 0  1  2 ... 30 31 32],
                    shape=[[6 4 6]
                           [3 6 4],
                           [2 2 2]],
                    dtype=int64)
    """

    assert all(
        isinstance(obj, JaggedArray) for obj in objs
    ), "all operands must be `JaggedArray`s"
    assert all(
        np.array_equal(objs[0].shape, other.shape) for other in objs[1:]
    ), "all shapes must be equal."
    return concatenate([obj.expand_dims(axis=-1) for obj in objs], axis=-1)
