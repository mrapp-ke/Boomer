from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport float32, uint8, uint32


cdef extern from "common/input/label_matrix.hpp" nogil:

    cdef cppclass ILabelMatrix:

        # Functions:

        uint32 getNumRows() const

        uint32 getNumCols() const

        bool isSparse() const


cdef extern from "common/input/label_matrix_row_wise.hpp" nogil:

    cdef cppclass IRowWiseLabelMatrix(ILabelMatrix):

        # Functions:

        float32 calculateLabelCardinality() const


cdef extern from "common/input/label_matrix_c_contiguous.hpp" nogil:

    cdef cppclass ICContiguousLabelMatrix(IRowWiseLabelMatrix):
        pass


    unique_ptr[ICContiguousLabelMatrix] createCContiguousLabelMatrix(uint32 numRows, uint32 numCols, const uint8* array)


cdef extern from "common/input/label_matrix_csr.hpp" nogil:

    cdef cppclass ICsrLabelMatrix(IRowWiseLabelMatrix):
        pass


    unique_ptr[ICsrLabelMatrix] createCsrLabelMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices,
                                                     uint32* colIndices)


cdef class LabelMatrix:

    # Functions:

    cdef ILabelMatrix* get_label_matrix_ptr(self)


cdef class RowWiseLabelMatrix(LabelMatrix):

    # Functions:

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self)


cdef class CContiguousLabelMatrix(RowWiseLabelMatrix):

    # Attributes:

    cdef const uint8[:, ::1] array

    cdef unique_ptr[ICContiguousLabelMatrix] label_matrix_ptr


cdef class CsrLabelMatrix(RowWiseLabelMatrix):

    # Attributes:

    cdef uint32[::1] row_indices

    cdef uint32[::1] col_indices

    cdef unique_ptr[ICsrLabelMatrix] label_matrix_ptr
