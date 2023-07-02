"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from cython.view cimport array as cvarray

from mlrl.common.cython._types cimport float64, uint8, uint32


cdef inline cvarray array_uint32(uint32* array, uint32 num_elements):
    """
    Creates and returns a new C-contiguous array of type `uint32`, shape `(num_elements)`, that takes ownership of a
    pre-allocated C array.

    :param array:           A pointer to an array of type `uint32`
    :param num_elements:    The number of elements in the array
    :return:                The array that has been created
    """
    cdef cvarray view = <uint32[:num_elements]>array
    view.free_data = True
    return view


cdef inline cvarray view_uint32(uint32* array, uint32 num_elements):
    """
    Creates and returns a new C-contiguous array of type `uint32`, shape `(num_elements)`, that provides access to a
    pre-allocated C array.

    :param array:           A pointer to an array of type `uint32`
    :param num_elements:    The number of elements in the array
    :return:                The array that has been created
    """
    cdef cvarray view = <uint32[:num_elements]>array
    view.free_data = False
    return view


cdef inline cvarray c_matrix_uint8(uint8* array, uint32 num_rows, uint32 num_cols):
    """
    Creates and returns a new C-contiguous array of type `uint8`, shape `(num_rows, num_cols)`, that takes ownership of
    a pre-allocated C array.

    :param array:       A pointer to an array of type `uint8`, shape `(num_rows, num_cols)`
    :param num_rows:    The number of rows in the array
    :param num_cols:    The number of columns in the array
    :return:            The array that has been created
    """
    cdef cvarray view = <uint8[:num_rows, :num_cols]>array
    view.free_data = True
    return view


cdef inline cvarray c_view_uint8(uint8* array, uint32 num_rows, uint32 num_cols):
    """
    Creates and returns a new C-contiguous array of type `uint8`, shape `(num_rows, num_cols)`, that provides access to
    a pre-allocated C array.

    :param array:       A pointer to an array of type `uint8`, shape `(num_rows, num_cols)`
    :param num_rows:    The number of rows in the array
    :param num_cols:    The number of columns in the array
    :return:            The array that has been created
    """
    cdef cvarray view = <uint8[:num_rows, :num_cols]>array
    view.free_data = False
    return view


cdef inline cvarray c_matrix_float64(float64* array, uint32 num_rows, uint32 num_cols):
    """
    Creates and returns a new C-contiguous array of type `float64`, shape `(num_rows, num_cols)`, that takes ownership
    of a pre-allocated C array.

    :param array:       A pointer to an array of type `float64`, shape `(num_rows * num_cols)`
    :param num_rows:    The number of rows in the array
    :param num_cols:    The number of columns in the array
    :return:            The array that has been created
    """
    cdef cvarray view = <float64[:num_rows, :num_cols]>array
    view.free_data = True
    return view


cdef inline cvarray c_view_float64(float64* array, uint32 num_rows, uint32 num_cols):
    """
    Creates and returns a new C-contiguous array of type `float64`, shape `(num_rows, num_cols)`, that provides access
    to a pre-allocated C array.

    :param array:       A pointer to an array of type `float64`, shape `(num_rows * num_cols)`
    :param num_rows:    The number of rows in the array
    :param num_cols:    The number of columns in the array
    :return:            The array that has been created
    """
    cdef cvarray view = <float64[:num_rows, :num_cols]>array
    view.free_data = False
    return view
