"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that provide access to the data that is provided for training.
"""
from boomer.common._tuples cimport IndexedFloat32, compareIndexedFloat32

from libc.stdlib cimport qsort, malloc

from libcpp.memory cimport make_shared

from cython.operator cimport dereference


cdef class LabelMatrix:
    """
    A wrapper for the abstract C++ class `AbstractLabelMatrix`.
    """
    pass


cdef class RandomAccessLabelMatrix(LabelMatrix):
    """
    A wrapper for the abstract C++ class `AbstractRandomAccessLabelMatrix`.
    """
    pass


cdef class DenseLabelMatrix(RandomAccessLabelMatrix):
    """
    A wrapper for the C++ class `DenseLabelMatrix`.
    """

    def __cinit__(self, const uint8[:, ::1] y):
        """
        :param y: An array of type `uint8`, shape `(num_examples, num_labels)`, representing the labels of the training
                  examples
        """
        cdef uint32 num_examples = y.shape[0]
        cdef uint32 num_labels = y.shape[1]
        self.label_matrix_ptr = <shared_ptr[AbstractLabelMatrix]>make_shared[DenseLabelMatrixImpl](num_examples,
                                                                                                   num_labels, &y[0, 0])
        self.num_examples = num_examples
        self.num_labels = num_labels


cdef class DokLabelMatrix(RandomAccessLabelMatrix):
    """
    A wrapper for the C++ class `DokLabelMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_labels, list[::1] rows):
        """
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        :param rows:            An array of type `list`, shape `(num_rows)`, storing a list for each example containing
                                the column indices of all non-zero labels
        """
        cdef shared_ptr[BinaryDokMatrix] dok_matrix_ptr
        cdef BinaryDokMatrix* dok_matrix = new BinaryDokMatrix()
        cdef uint32 num_rows = rows.shape[0]
        cdef list col_indices
        cdef uint32 r, c

        for r in range(num_rows):
            col_indices = rows[r]

            for c in col_indices:
                dok_matrix.addValue(r, c)

        dok_matrix_ptr.reset(dok_matrix)
        self.label_matrix_ptr = <shared_ptr[AbstractLabelMatrix]>make_shared[DokLabelMatrixImpl](num_examples,
                                                                                                 num_labels,
                                                                                                 dok_matrix_ptr)
        self.num_examples = num_examples
        self.num_labels = num_labels


cdef class FeatureMatrix:
    """
    A base class for all classes that provide column-wise access to the feature values of the training examples.
    """

    cdef void fetch_sorted_feature_values(self, uint32 feature_index, IndexedFloat32Array* indexed_array) nogil:
        """
        Fetches the indices of the training examples, as well as their feature values, for a specific feature, sorts
        them in ascending order by the feature values and stores the in a given struct of type `IndexedFloat32Array`.

        :param feature_index:   The index of the feature
        :param indexed_array:   A pointer to a struct of type `IndexedFloat32Array`, which should be used to store the
                                indices
        """
        pass


cdef class DenseFeatureMatrix(FeatureMatrix):
    """
    Implements column-wise access to the feature values of the training examples based on a dense feature matrix.

    The feature matrix must be given as a dense Fortran-contiguous array.
    """

    def __cinit__(self, const float32[::1, :] x):
        """
        :param x: An array of type `float32`, shape `(num_examples, num_features)`, representing the feature values of
                  the training examples
        """
        self.num_examples = x.shape[0]
        self.num_features = x.shape[1]
        self.x = x

    cdef void fetch_sorted_feature_values(self, uint32 feature_index, IndexedFloat32Array* indexed_array) nogil:
        # Class members
        cdef const float32[::1, :] x = self.x
        # The number of elements to be returned
        cdef uint32 num_elements = x.shape[0]
        # The array that stores the indices
        cdef IndexedFloat32* sorted_array = <IndexedFloat32*>malloc(num_elements * sizeof(IndexedFloat32))
        # Temporary variables
        cdef uint32 i

        for i in range(num_elements):
            sorted_array[i].index = i
            sorted_array[i].value = x[i, feature_index]

        qsort(sorted_array, num_elements, sizeof(IndexedFloat32), &compareIndexedFloat32)

        # Update the given struct...
        indexed_array.numElements = num_elements
        indexed_array.data = sorted_array


cdef class CscFeatureMatrix(FeatureMatrix):
    """
    Implements column-wise access to the feature values of the training examples based on a sparse feature matrix.

    The feature matrix must be given in compressed sparse column (CSC) format.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] x_data,
                  const uint32[::1] x_row_indices, const uint32[::1] x_col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param x_data:          An array of type `float32`, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the training examples
        :param x_row_indices:   An array of type `uint32`, shape `(num_non_zero_feature_values)`, representing the
                                row-indices of the examples, the values in `x_data` correspond to
        :param x_col_indices:   An array of type `uint32`, shape `(num_features + 1)`, representing the indices of the
                                first element in `x_data` and `x_row_indices` that corresponds to a certain feature. The
                                index at the last position is equal to `num_non_zero_feature_values`
        """
        self.num_examples = num_examples
        self.num_features = num_features
        self.x_data = x_data
        self.x_row_indices = x_row_indices
        self.x_col_indices = x_col_indices

    cdef void fetch_sorted_feature_values(self, uint32 feature_index, IndexedFloat32Array* indexed_array) nogil:
        # Class members
        cdef const float32[::1] x_data = self.x_data
        cdef const uint32[::1] x_row_indices = self.x_row_indices
        cdef const uint32[::1] x_col_indices = self.x_col_indices
        # The index of the first element in `x_data` and `x_row_indices` that corresponds to the given feature index
        cdef uint32 start = x_col_indices[feature_index]
        # The index of the last element in `x_data` and `x_row_indices` that corresponds to the given feature index
        cdef uint32 end = x_col_indices[feature_index + 1]
        # The number of elements to be returned
        cdef uint32 num_elements = end - start
        # The array that stores the indices
        cdef IndexedFloat32* sorted_array = NULL
        # Temporary variables
        cdef uint32 i, j

        if num_elements > 0:
            sorted_array = <IndexedFloat32*>malloc(num_elements * sizeof(IndexedFloat32))
            i = 0

            for j in range(start, end):
                sorted_array[i].index = x_row_indices[j]
                sorted_array[i].value = x_data[j]
                i += 1

            qsort(sorted_array, num_elements, sizeof(IndexedFloat32), &compareIndexedFloat32)

        # Update the given struct...
        indexed_array.numElements = num_elements
        indexed_array.data = sorted_array
