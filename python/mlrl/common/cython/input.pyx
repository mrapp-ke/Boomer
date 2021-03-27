"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_unique, make_shared
from libcpp.utility cimport move


cdef class LabelMatrix:
    """
    A wrapper for the pure virtual C++ class `ILabelMatrix`.
    """
    pass


cdef class RandomAccessLabelMatrix(LabelMatrix):
    """
    A wrapper for the pure virtual C++ class `IRandomAccessLabelMatrix`.
    """
    pass


cdef class CContiguousLabelMatrix(RandomAccessLabelMatrix):
    """
    A wrapper for the C++ class `CContiguousLabelMatrix`.
    """

    def __cinit__(self, uint8[:, ::1] array):
        """
        :param array: A C-contiguous array of type `uint8`, shape `(num_examples, num_labels)`, that stores the labels
                      of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_labels = array.shape[1]
        self.label_matrix_ptr = <shared_ptr[ILabelMatrix]>make_shared[CContiguousLabelMatrixImpl](num_examples,
                                                                                                  num_labels,
                                                                                                  &array[0, 0])


cdef class DokLabelMatrix(RandomAccessLabelMatrix):
    """
    A wrapper for the C++ class `DokLabelMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_labels, list[::1] rows):
        """
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        :param rows:            An array of type `list`, shape `(num_rows)`, that stores a list for each example, which
                                contains the column indices of all non-zero labels
        """
        cdef unique_ptr[DokLabelMatrixImpl] ptr = make_unique[DokLabelMatrixImpl](num_examples, num_labels)
        cdef uint32 num_rows = rows.shape[0]
        cdef list col_indices
        cdef uint32 r, c

        for r in range(num_rows):
            col_indices = rows[r]

            for c in col_indices:
                ptr.get().setValue(r, c)

        self.label_matrix_ptr = <shared_ptr[ILabelMatrix]>move(ptr)


cdef class FeatureMatrix:
    """
    A wrapper for the pure virtual C++ class `IFeatureMatrix`.
    """
    pass


cdef class FortranContiguousFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `FortranContiguousFeatureMatrix`.
    """

    def __cinit__(self, float32[::1, :] array):
        """
        :param array: A Fortran-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores
                      the feature values of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = <shared_ptr[IFeatureMatrix]>make_shared[FortranContiguousFeatureMatrixImpl](
            num_examples, num_features, &array[0, 0])


cdef class CscFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `CscFeatureMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] data,
                  const uint32[::1] row_indices, const uint32[::1] col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param data:            An array of type `float32`, shape `(num_non_zero_values)`, that stores all non-zero
                                feature values
        :param row_indices:     An array of type `uint32`, shape `(num_non_zero_values)`, that stores the row-indices,
                                the values in `data` correspond to
        :param col_indices:     An array of type `uint32`, shape `(num_features + 1)`, that stores the indices of the
                                first element in `data` and `row_indices` that corresponds to a certain feature. The
                                index at the last position is equal to `num_non_zero_values`
        """
        self.feature_matrix_ptr = <shared_ptr[IFeatureMatrix]>make_shared[CscFeatureMatrixImpl](num_examples,
                                                                                                num_features, &data[0],
                                                                                                &row_indices[0],
                                                                                                &col_indices[0])


cdef class CContiguousFeatureMatrix:
    """
    A wrapper for the C++ class `CContiguousFeatureMatrix`.
    """

    def __cinit__(self, float32[:, ::1] array):
        """
        :param array: A C-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores the
                      feature values of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = make_shared[CContiguousFeatureMatrixImpl](num_examples, num_features, &array[0, 0])


cdef class CsrFeatureMatrix:
    """
    A wrapper for the C++ class `CsrFeatureMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] data,
                  const uint32[::1] row_indices, const uint32[::1] col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param data:            An array of type `float32`, shape `(num_non_zero_values)`, that stores all non-zero
                                feature values
        :param row_indices:     An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `data` and `col_indices` that corresponds to a certain example. The
                                index at the last position is equal to `num_non_zero_values`
        :param col_indices:     An array of type `uint32`, shape `(num_non_zero_values)`, that stores the
                                column-indices, the values in `data` correspond to
        """
        self.feature_matrix_ptr = make_shared[CsrFeatureMatrixImpl](num_examples, num_features, &data[0],
                                                                    &row_indices[0], &col_indices[0])


cdef class NominalFeatureMask:
    """
    A wrapper for the pure virtual C++ class `INominalFeatureMask`.
    """
    pass


cdef class DokNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the C++ class `DokNominalFeatureMask`.
    """

    def __cinit__(self, list nominal_feature_indices):
        """
        :param nominal_feature_indices: A list which contains the indices of all nominal features or None, if no nominal
                                        features are available
        """
        cdef uint32 num_nominal_features = 0 if nominal_feature_indices is None else len(nominal_feature_indices)
        cdef unique_ptr[DokNominalFeatureMaskImpl] ptr = make_unique[DokNominalFeatureMaskImpl]()
        cdef uint32 i

        if num_nominal_features > 0:
            for i in nominal_feature_indices:
                ptr.get().setNominal(i)

        self.nominal_feature_mask_ptr = <shared_ptr[INominalFeatureMask]>move(ptr)


cdef class EqualNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the C++ class `EqualNominalFeatureMask`.
    """

    def __cinit__(self, bint nominal):
        """
        :param nominal: True, if all features are nominal, false, if all features are not nominal
        """
        self.nominal_feature_mask_ptr = <shared_ptr[INominalFeatureMask]>make_shared[EqualNominalFeatureMaskImpl](
            nominal)
