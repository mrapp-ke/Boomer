"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._types cimport uint32


cdef extern from "common/data/view_c_contiguous.hpp" nogil:

    cdef cppclass CContiguousView[T]:

        # Constructors:

        CContiguousView(uint32 numRows, uint32 numCols, T* array)
