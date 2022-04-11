"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class FeatureMatrix:
    """
    A feature matrix.
    """

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        pass

    def get_num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        :return The number of rows
        """
        return self.get_feature_matrix_ptr().getNumRows()

    def get_num_cols(self) -> int:
        """
        Returns the number of columns in the matrix.

        :return The number of columns
        """
        return self.get_feature_matrix_ptr().getNumCols()

    def is_sparse(self) -> bool:
        """
        Returns whether the feature matrix is sparse or not.

        :return: True, if the feature matrix is sparse, False otherwise
        """
        return self.get_feature_matrix_ptr().isSparse()


cdef class ColumnWiseFeatureMatrix(FeatureMatrix):
    """
    A feature matrix that provides column-wise access to the feature values of examples.
    """

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        pass


cdef class FortranContiguousFeatureMatrix(ColumnWiseFeatureMatrix):
    """
    A feature matrix that provides column-wise access to the feature values of examples that are stored in a
    Fortran-contiguous array.
    """

    def __cinit__(self, const float32[::1, :] array not None):
        """
        :param array: A Fortran-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores
                      the feature values of the training examples
        """
        self.array = array
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = createFortranContiguousFeatureMatrix(num_examples, num_features, &array[0, 0])

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()


cdef class CscFeatureMatrix(ColumnWiseFeatureMatrix):
    """
    A feature matrix that provides column-wise access to the feature values of examples that are stored in a sparse
    matrix in the compressed sparse column (CSC) format.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] data not None,
                  uint32[::1] row_indices not None, uint32[::1] col_indices not None):
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
        self.data = data
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.feature_matrix_ptr = createCscFeatureMatrix(num_examples, num_features, &data[0], &row_indices[0],
                                                         &col_indices[0])

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IColumnWiseFeatureMatrix* get_column_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()


cdef class RowWiseFeatureMatrix(FeatureMatrix):
    """
    A feature matrix that provides row-wise access to the feature values of examples.
    """

    cdef IRowWiseFeatureMatrix* get_row_wise_feature_matrix_ptr(self):
        pass


cdef class CContiguousFeatureMatrix(RowWiseFeatureMatrix):
    """
    A feature matrix that provides row-wise access to the feature values of examples that are stored in a C-contiguous
    array.
    """

    def __cinit__(self, const float32[:, ::1] array not None):
        """
        :param array: A C-contiguous array of type `float32`, shape `(num_examples, num_features)`, that stores the
                      feature values of the training examples
        """
        self.array = array
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_features = array.shape[1]
        self.feature_matrix_ptr = createCContiguousFeatureMatrix(num_examples, num_features, &array[0, 0])

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IRowWiseFeatureMatrix* get_row_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()


cdef class CsrFeatureMatrix(RowWiseFeatureMatrix):
    """
    A feature matrix that provides row-wise access to the feature values of examples that are stored in a sparse matrix
    in the compressed sparse row (CSR) format.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_features, const float32[::1] data not None,
                  uint32[::1] row_indices not None, uint32[::1] col_indices not None):
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
        self.data = data
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.feature_matrix_ptr = createCsrFeatureMatrix(num_examples, num_features, &data[0], &row_indices[0],
                                                         &col_indices[0])

    cdef IFeatureMatrix* get_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()

    cdef IRowWiseFeatureMatrix* get_row_wise_feature_matrix_ptr(self):
        return self.feature_matrix_ptr.get()
