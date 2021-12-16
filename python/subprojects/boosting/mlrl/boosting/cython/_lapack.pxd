"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from scipy.linalg.cython_lapack cimport dsysv

from libcpp.memory cimport unique_ptr, make_unique


cdef extern from "boosting/math/lapack.hpp" namespace "boosting" nogil:

    ctypedef void (*dsysv_t)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork, int* info)

    cdef cppclass Lapack:

        # Constructors:

        Lapack(dsysv_t dsysvFunction) except +


cdef inline unique_ptr[Lapack] init_lapack():
    """
    Creates a new wrapper for executing different LAPACK routines.

    :return: An unique pointer to an object of type `Lapack` that allows to execute different LAPACK routines
    """
    return make_unique[Lapack](dsysv)
