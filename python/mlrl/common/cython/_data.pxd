"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython._types cimport uint32


cdef extern from "common/data/indexed_value.hpp" nogil:

    cdef cppclass IndexedValue[T]:

        # Attributes:

        uint32 index

        T value


cdef extern from "common/data/view_c_contiguous.hpp" nogil:

    cdef cppclass CContiguousView[T]:

        # Constructors:

        CContiguousView(uint32 numRows, uint32 numCols, T* array)
