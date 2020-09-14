from boomer.common._arrays cimport uint8, uint32, float32
from boomer.common._tuples cimport IndexedFloat32Array
from boomer.common._sparse cimport BinaryDokMatrix

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/input_data.h" nogil:

    cdef cppclass AbstractLabelMatrix:
        pass


    cdef cppclass AbstractRandomAccessLabelMatrix(AbstractLabelMatrix):
        pass


    cdef cppclass DenseLabelMatrixImpl(AbstractRandomAccessLabelMatrix):

        # Constructors:

        DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, uint8* y) except +


    cdef cppclass DokLabelMatrixImpl(AbstractRandomAccessLabelMatrix):

        # Constructors:

        DokLabelMatrixImpl(uint32 numExamples, uint32 numLabels, shared_ptr[BinaryDokMatrix] dokMatrix) except +


cdef class LabelMatrix:

    # Attributes:

    cdef shared_ptr[AbstractLabelMatrix] label_matrix_ptr

    cdef readonly uint32 num_examples

    cdef readonly uint32 num_labels


cdef class RandomAccessLabelMatrix(LabelMatrix):
    pass


cdef class DenseLabelMatrix(RandomAccessLabelMatrix):
    pass


cdef class DokLabelMatrix(RandomAccessLabelMatrix):
    pass


cdef class FeatureMatrix:

    # Attributes:

    cdef readonly uint32 num_examples

    cdef readonly uint32 num_features

    # Functions:

    cdef void fetch_sorted_feature_values(self, uint32 feature_index, IndexedFloat32Array* indexed_array) nogil


cdef class DenseFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef const float32[::1, :] x

    # Functions:

    cdef void fetch_sorted_feature_values(self, uint32 feature_index, IndexedFloat32Array* indexed_array) nogil


cdef class CscFeatureMatrix(FeatureMatrix):

    # Attributes:

    cdef const float32[::1] x_data

    cdef const uint32[::1] x_row_indices

    cdef const uint32[::1] x_col_indices

    # Functions:

    cdef void fetch_sorted_feature_values(self, uint32 feature_index, IndexedFloat32Array* indexed_array) nogil
