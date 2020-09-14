"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a function that allows to create a wrapper for executing different LAPACK routines.

The function pointers to the different LAPACK routines are initialized such that they refer to the functions provided by
scipy.
"""
from scipy.linalg.cython_lapack cimport dsysv


cdef extern from "cpp/lapack.h" nogil:

    ctypedef void (*dsysv_t)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork, int* info)

    cdef cppclass Lapack:

        # Constructors:

        Lapack(dsysv_t dsysvFunction) except +


cdef inline Lapack* init_lapack():
    """
    Creates a new wrapper for executing different LAPACK routines.

    :return: A pointer to an object of type `Lapack` that allows to execute different LAPACK routines
    """
    return new Lapack(dsysv)
