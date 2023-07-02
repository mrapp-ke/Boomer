"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for handling arrays.
"""
import numpy as np

from scipy.sparse import issparse


def enforce_dense(a, order: str, dtype) -> np.ndarray:
    """
    Converts a given array into a `np.ndarray`, if necessary, and enforces a specific memory layout and data type to be
    used.

    :param a:       A `np.ndarray` or `scipy.sparse.matrix` to be converted
    :param order:   The memory layout to be used. Must be `C` or `F`
    :param dtype:   The data type to be used
    :return:        A `np.ndarray` that uses the given memory layout and data type
    """
    if issparse(a):
        return np.require(a.toarray(order=order), dtype=dtype)
    else:
        return np.require(a, dtype=dtype, requirements=[order])


def enforce_2d(a: np.ndarray) -> np.ndarray:
    """
    Converts a given `np.ndarray` into a two-dimensional array if it is one-dimensional.

    :param a:   A `np.ndarray` to be converted
    :return:    A `np.ndarray` with at least two dimensions
    """
    if a.ndim == 1:
        return np.expand_dims(a, axis=1)
    else:
        return a
