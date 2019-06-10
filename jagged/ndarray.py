import numpy as np

from .core import JaggedArray


def jagged_to_array(jarr: JaggedArray, copy=True) -> np.ndarray:
    """ Create a numpy array from a jagged array. """
    if jarr.is_jagged:
        msg = (
            "Cannot create a smoothe array from jagged. Try `to_iliffe` or `to_masked`."
        )
        raise ValueError(msg)
    else:
        return np.array(jarr, copy=copy)


def array_to_jagged(arr: np.ndarray, copy=True) -> JaggedArray:
    """ Create a JaggedArray from a numpy array. """

    return JaggedArray(
        arr.shape, buffer=arr.data, strides=arr.strides[1:], dtype=arr.dtype
    )
