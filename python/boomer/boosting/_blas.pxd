"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a function that allows to create a wrapper for executing different BLAS routines.

The function pointers to the different BLAS routines are initialized such that they refer to the functions provided by
scipy.
"""
from scipy.linalg.cython_blas cimport ddot, dspmv


cdef extern from "cpp/blas.h" nogil:

    ctypedef double (*ddot_t)(int* n, double* dx, int* incx, double* dy, int* incy)

    ctypedef void (*dspmv_t)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta, double* y, int* incy)

    cdef cppclass Blas:

        # Constructors:

        Blas(ddot_t ddotFunction, dspmv_t dspmvFunction) except +


cdef inline Blas* init_blas():
    """
    Creates a new wrapper for executing different BLAS routines.

    :return: A pointer to an object of type `Blas` that allows to execute different BLAS routines
    """
    return new Blas(ddot, dspmv)
