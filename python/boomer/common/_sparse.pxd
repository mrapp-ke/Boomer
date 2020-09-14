"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides implementations of sparse matrices.
"""
from boomer.common._arrays cimport uint32


cdef extern from "cpp/sparse.h" nogil:

    cdef cppclass BinaryDokMatrix:

        # Constructors:

        BinaryDokMatrix() except +

        # Functions:

        void addValue(uint32 row, uint32 column)
