"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move
from libcpp.memory cimport make_unique

SERIALIZATION_VERSION = 1


cdef class LabelMatrix:
    """
    A wrapper for the pure virtual C++ class `ILabelMatrix`.
    """

    def get_num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        :return: The number of rows
        """
        return self.label_matrix_ptr.get().getNumRows()

    def get_num_cols(self) -> int:
        """
        Returns the number of columns in the matrix.

        :return: The number of columns
        """
        return self.label_matrix_ptr.get().getNumCols()

    def is_sparse(self) -> bool:
        """
        Returns whether the label matrix is sparse or not.

        :return: True, if the label matrix is sparse, False otherwise
        """
        return False

    def calculate_label_cardinality(self) -> float:
        """
        Calculates and returns the label cardinality, i.e., the average number of relevant labels per example.

        :return: The label cardinality
        """
        return self.label_matrix_ptr.get().calculateLabelCardinality()


cdef class CContiguousLabelMatrix(LabelMatrix):
    """
    A wrapper for the C++ class `CContiguousLabelMatrix`.
    """

    def __cinit__(self, const uint8[:, ::1] array):
        """
        :param array: A C-contiguous array of type `uint8`, shape `(num_examples, num_labels)`, that stores the labels
                      of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_labels = array.shape[1]
        self.label_matrix_ptr = <unique_ptr[ILabelMatrix]>make_unique[CContiguousLabelMatrixImpl](num_examples,
                                                                                                  num_labels,
                                                                                                  &array[0, 0])


cdef class CsrLabelMatrix(LabelMatrix):
    """
    A wrapper for the C++ class `CsrLabelMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_labels, uint32[::1] row_indices, uint32[::1] col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        :param row_indices:     An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `col_indices` that corresponds to a certain example. The index at the
                                last position is equal to `num_non_zero_values`
        :param col_indices:     An array of type `uint32`, shape `(num_non_zero_values)`, that stores the
                                column-indices, the relevant labels correspond to
        """
        self.label_matrix_ptr = <unique_ptr[ILabelMatrix]>make_unique[CsrLabelMatrixImpl](num_examples, num_labels,
                                                                                          &row_indices[0],
                                                                                          &col_indices[0])

    def is_sparse(self) -> bool:
        return True


cdef class FeatureMatrix:
    """
    A wrapper for the pure virtual C++ class `IFeatureMatrix`.
    """

    def get_num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        :return The number of rows
        """
        return self.feature_matrix_ptr.get().getNumRows()

    def get_num_cols(self) -> int:
        """
        Returns the number of columns in the matrix.

        :return The number of columns
        """
        return self.feature_matrix_ptr.get().getNumCols()

    def is_sparse(self) -> bool:
        """
        Returns whether the feature matrix is sparse or not.

        :return: True, if the feature matrix is sparse, False otherwise
        """
        return False


cdef class FortranContiguousFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `FortranContiguousFeatureMatrix`.
    """

    def __cinit__(self, const float32[::1, :] array):
        """
        :param array: A Fortran-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores
                      the feature values of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = <unique_ptr[IFeatureMatrix]>make_unique[FortranContiguousFeatureMatrixImpl](
            num_examples, num_features, &array[0, 0])


cdef class CscFeatureMatrix(FeatureMatrix):
    """
    A wrapper for the C++ class `CscFeatureMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] data, uint32[::1] row_indices,
                  uint32[::1] col_indices):
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
        self.feature_matrix_ptr = <unique_ptr[IFeatureMatrix]>make_unique[CscFeatureMatrixImpl](num_examples,
                                                                                                num_features, &data[0],
                                                                                                &row_indices[0],
                                                                                                &col_indices[0])

    def is_sparse(self) -> bool:
        return True


cdef class CContiguousFeatureMatrix:
    """
    A wrapper for the C++ class `CContiguousFeatureMatrix`.
    """

    def __cinit__(self, const float32[:, ::1] array):
        """
        :param array: A C-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores the
                      feature values of the training examples
        """
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = make_unique[CContiguousFeatureMatrixImpl](num_examples, num_features, &array[0, 0])


cdef class CsrFeatureMatrix:
    """
    A wrapper for the C++ class `CsrFeatureMatrix`.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] data, uint32[::1] row_indices,
                  uint32[::1] col_indices):
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
        self.feature_matrix_ptr = make_unique[CsrFeatureMatrixImpl](num_examples, num_features, &data[0],
                                                                    &row_indices[0], &col_indices[0])

    def is_sparse(self) -> bool:
        return True


cdef class NominalFeatureMask:
    """
    A wrapper for the pure virtual C++ class `INominalFeatureMask`.
    """
    pass


cdef class BitNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the C++ class `BitNominalFeatureMask`.
    """

    def __cinit__(self, uint32 num_features, list nominal_feature_indices):
        """
        :param num_features:            The total number of available features
        :param nominal_feature_indices: A list which contains the indices of all nominal features
        """
        cdef unique_ptr[BitNominalFeatureMaskImpl] ptr = make_unique[BitNominalFeatureMaskImpl](num_features)
        cdef uint32 i

        for i in nominal_feature_indices:
            ptr.get().setNominal(i)

        self.nominal_feature_mask_ptr = <unique_ptr[INominalFeatureMask]>move(ptr)


cdef class EqualNominalFeatureMask(NominalFeatureMask):
    """
    A wrapper for the C++ class `EqualNominalFeatureMask`.
    """

    def __cinit__(self, bint nominal):
        """
        :param nominal: True, if all features are nominal, false, if all features are not nominal
        """
        self.nominal_feature_mask_ptr = <unique_ptr[INominalFeatureMask]>make_unique[EqualNominalFeatureMaskImpl](
            nominal)


cdef class LabelVectorSet:
    """
    A wrapper for the C++ class `LabelVectorSet`.
    """

    def __cinit__(self):
        self.label_vector_set_ptr = make_unique[LabelVectorSetImpl]()

    @classmethod
    def create(cls, LabelMatrix label_matrix):
        cdef ILabelMatrix* label_matrix_ptr = label_matrix.label_matrix_ptr.get()
        cdef uint32 num_rows = label_matrix_ptr.getNumRows()
        cdef uint32 num_cols = label_matrix_ptr.getNumCols()
        cdef unique_ptr[LabelVectorSetImpl] label_vector_set_ptr = make_unique[LabelVectorSetImpl]()
        cdef unique_ptr[LabelVector] label_vector_ptr
        cdef uint32 i

        for i in range(num_rows):
            label_vector_ptr = label_matrix_ptr.createLabelVector(i)
            label_vector_set_ptr.get().addLabelVector(move(label_vector_ptr))

        cdef LabelVectorSet label_vector_set = LabelVectorSet.__new__(LabelVectorSet)
        label_vector_set.label_vector_set_ptr = move(label_vector_set_ptr)
        return label_vector_set

    def __reduce__(self):
        cdef LabelVectorSetSerializer serializer = LabelVectorSetSerializer.__new__(LabelVectorSetSerializer)
        cdef object state = serializer.serialize(self)
        return (LabelVectorSet, (), state)

    def __setstate__(self, state):
        cdef LabelVectorSetSerializer serializer = LabelVectorSetSerializer.__new__(LabelVectorSetSerializer)
        serializer.deserialize(self, state)


cdef inline unique_ptr[LabelVector] __create_label_vector(list state):
    cdef uint32 num_elements = len(state)
    cdef unique_ptr[LabelVector] label_vector_ptr = make_unique[LabelVector](num_elements)
    cdef LabelVector.index_iterator iterator = label_vector_ptr.get().indices_begin()
    cdef uint32 i, label_index

    for i in range(num_elements):
        label_index = state[i]
        iterator[i] = label_index

    return move(label_vector_ptr)


cdef class LabelVectorSetSerializer:
    """
    Allows to serialize and deserialize the label vectors that are stored by a `LabelVectorSet`.
    """

    cdef __visit_label_vector(self, const LabelVector& label_vector):
        cdef list label_vector_state = []
        cdef uint32 num_elements = label_vector.getNumElements()
        cdef LabelVector.index_const_iterator iterator = label_vector.indices_cbegin()
        cdef uint32 i, label_index

        for i in range(num_elements):
            label_index = iterator[i]
            label_vector_state.append(label_index)

        self.state.append(label_vector_state)

    def serialize(self, LabelVectorSet label_vector_set not None):
        """
        Creates and returns a state, which may be serialized using Python's pickle mechanism, from the label vectors
        that are stored by a given `LabelVectorSet`.

        :param label_vector_set:    The set that stores the label vectors to be serialized
        :return:                    The state that has been created
        """
        self.state = []
        cdef LabelVectorSetImpl* label_vector_set_ptr = label_vector_set.label_vector_set_ptr.get()
        label_vector_set_ptr.visit(wrapLabelVectorVisitor(<void*>self,
                                                          <LabelVectorCythonVisitor>self.__visit_label_vector))
        return (SERIALIZATION_VERSION, self.state)

    def deserialize(self, LabelVectorSet label_vector_set not None, object state not None):
        """
        Deserializes the label vectors that are stored by a given state and adds them to a `LabelVectorSet`.

        :param label_vector_set:    The set, the deserialized rules should be added to
        :param state:               A state that has previously been created via the function `serialize`
        """
        cdef int version = state[0]

        if version != SERIALIZATION_VERSION:
            raise AssertionError(
                'Version of the serialized LabelVectorSet is ' + str(version) + ', expected ' + str(SERIALIZATION_VERSION))

        cdef list label_vector_list = state[1]
        cdef uint32 num_label_vectors = len(label_vector_list)
        cdef LabelVectorSetImpl* label_vector_set_ptr = label_vector_set.label_vector_set_ptr.get()
        cdef list label_vector_state
        cdef uint32 i

        for i in range(num_label_vectors):
            label_vector_state = label_vector_list[i]
            label_vector_set_ptr.addLabelVector(move(__create_label_vector(label_vector_state)))
