"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython._types cimport uint8, uint32, float64

from cython.view cimport array as cvarray

DEF MODE_C_CONTIGUOUS = 'c'
DEF FORMAT_UINT8 = 'B'
DEF FORMAT_FLOAT64 = 'd'


cdef inline cvarray c_matrix_uint8(uint32 num_rows, uint32 num_cols):
    """
    Creates and returns a new C-contiguous array of type `uint8`, shape `(num_rows, num_cols)`.

    :param num_rows:    The number of rows in the array
    :param num_cols:    The number of columns in the array
    :return:            The array that has been created
    """
    cdef tuple shape = tuple([num_rows, num_cols])
    cdef int itemsize = sizeof(uint8)
    cdef cvarray array = cvarray(shape, itemsize, FORMAT_UINT8, MODE_C_CONTIGUOUS)
    return array


cdef inline cvarray c_matrix_float64(uint32 num_rows, uint32 num_cols):
    """
    Creates and returns a new C-contiguous array of type `float64`, shape `(num_rows, num_cols)`.

    :param num_rows:    The number of rows in the array
    :param num_cols:    The number of columns in the array
    :return:            The array that has been created
    """
    cdef tuple shape = tuple([num_rows, num_cols])
    cdef int itemsize = sizeof(float64)
    cdef cvarray array = cvarray(shape, itemsize, FORMAT_FLOAT64, MODE_C_CONTIGUOUS)
    return array
