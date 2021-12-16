"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from scipy.linalg.cython_blas cimport ddot, dspmv

from libcpp.memory cimport unique_ptr, make_unique


cdef extern from "boosting/math/blas.hpp" namespace "boosting" nogil:

    ctypedef double (*ddot_t)(int* n, double* dx, int* incx, double* dy, int* incy)

    ctypedef void (*dspmv_t)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta, double* y, int* incy)

    cdef cppclass Blas:

        # Constructors:

        Blas(ddot_t ddotFunction, dspmv_t dspmvFunction) except +


cdef inline unique_ptr[Blas] init_blas():
    """
    Creates a new wrapper for executing different BLAS routines.

    :return: An unique pointer to an object of type `Blas` that allows to execute different BLAS routines
    """
    return make_unique[Blas](ddot, dspmv)
