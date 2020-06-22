# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides type definitions and utility functions for creating arrays.
"""
from cython.view cimport array as cvarray

cimport numpy as npc

ctypedef Py_ssize_t intp
ctypedef npc.uint8_t uint8
ctypedef npc.uint32_t uint32
ctypedef npc.float32_t float32
ctypedef npc.float64_t float64

DEF MODE_C_CONTIGUOUS = 'c'
DEF MODE_FORTRAN_CONTIGUOUS = 'fortran'

IF UNAME_SYSNAME == 'Windows':
    DEF FORMAT_UINT32 = 'I'
    DEF FORMAT_INTP = 'q'
    DEF FORMAT_FLOAT64 = 'd'
ELSE:
    DEF FORMAT_UINT32 = 'I'
    DEF FORMAT_INTP = 'l'
    DEF FORMAT_FLOAT64 = 'd'

cdef inline cvarray array_intp(intp num_elements):
    """
    Creates and returns a new C-contiguous array of dtype `intp`, shape `(num_elements)`.

    :param num_elements:    The number of elements in the array
    :return:                The array that has been created
    """
    cdef tuple shape = tuple([num_elements])
    cdef intp itemsize = sizeof(intp)
    cdef cvarray array = cvarray(shape, itemsize, FORMAT_INTP, MODE_C_CONTIGUOUS)
    return array

cdef inline cvarray array_uint32(intp num_elements):
    """
    Creates and returns a new C-contiguous array of dtype `uint32`, shape `(num_elements)`.

    :param num_elements:    The number of elements in the array
    :return:                The array that has been created
    """
    cdef tuple shape = tuple([num_elements])
    cdef intp itemsize = sizeof(uint32)
    cdef cvarray array = cvarray(shape, itemsize, FORMAT_UINT32, MODE_C_CONTIGUOUS)
    return array

cdef inline cvarray array_float64(intp num_elements):
    """
    Creates and returns a new C-contiguous array of dtype `float64`, shape `(num_elements)`.

    :param num_elements:    The number of elements in the array
    :return:                The array that has been created
    """
    cdef tuple shape = tuple([num_elements])
    cdef intp itemsize = sizeof(float64)
    cdef cvarray array = cvarray(shape, itemsize, FORMAT_FLOAT64, MODE_C_CONTIGUOUS)
    return array

cdef inline cvarray matrix_intp(intp num_rows, intp num_cols):
    """
    Creates and returns a new Fortran-contiguous array of dtype `intp`, shape `(num_rows, num_cols)`.

    :param num_rows:    The number of rows in the array
    :param num_cols:    The number of columns in the array
    :return:            The array that has been created
    """
    cdef tuple shape = tuple([num_rows, num_cols])
    cdef intp itemsize = sizeof(intp)
    cdef cvarray array = cvarray(shape, itemsize, FORMAT_INTP, MODE_FORTRAN_CONTIGUOUS)
    return array

cdef inline cvarray matrix_float64(intp num_rows, num_cols):
    """
    Creates and returns a new Fortran-contiguous array of dtype `float64`, shape `(num_rows, num_cols)`.

    :param num_rows:    The number of rows in the array
    :param num_cols:    The number of columns in the array
    :return:            The array that has been created
    """
    cdef tuple shape = tuple([num_rows, num_cols])
    cdef intp itemsize = sizeof(float64)
    cdef cvarray array = cvarray(shape, itemsize, FORMAT_FLOAT64, MODE_FORTRAN_CONTIGUOUS)
    return array
