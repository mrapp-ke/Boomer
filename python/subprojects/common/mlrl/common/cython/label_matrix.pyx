"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class LabelMatrix:
    """
    A label matrix.
    """

    cdef ILabelMatrix* get_label_matrix_ptr(self):
        pass

    def get_num_rows(self) -> int:
        """
        Returns the number of rows in the matrix.

        :return The number of rows
        """
        return self.get_label_matrix_ptr().getNumRows()

    def get_num_cols(self) -> int:
        """
        Returns the number of columns in the matrix.

        :return The number of columns
        """
        return self.get_label_matrix_ptr().getNumCols()

    def is_sparse(self) -> bool:
        """
        Returns whether the label matrix is sparse or not.

        :return: True, if the label matrix is sparse, False otherwise
        """
        return self.get_label_matrix_ptr().isSparse()


cdef class RowWiseLabelMatrix(LabelMatrix):
    """
    A label matrix that provides row-wise access to the labels of examples.
    """

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        pass

    def calculate_label_cardinality(self) -> float:
        return self.get_row_wise_label_matrix_ptr().calculateLabelCardinality()


cdef class CContiguousLabelMatrix(RowWiseLabelMatrix):
    """
    A label matrix that provides row-wise access to the labels of examples that are stored in a C-contiguous array.
    """

    def __cinit__(self, const uint8[:, ::1] array not None):
        """
        :param array: A C-contiguous array of type `uint8`, shape `(num_examples, num_labels)`, that stores the labels
                      of the training examples
        """
        self.array = array
        cdef uint32 num_examples = array.shape[0]
        cdef uint32 num_labels = array.shape[1]
        self.label_matrix_ptr = createCContiguousLabelMatrix(num_examples, num_labels, &array[0, 0])

    cdef ILabelMatrix* get_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()


cdef class CsrLabelMatrix(RowWiseLabelMatrix):
    """
    A label matrix that provides row-wise access to the labels of examples that are stored in a sparse matrix in the
    compressed sparse row (CSR) format.
    """

    def __cinit__(self, uint32 num_examples, uint32 num_labels, uint32[::1] row_indices not None,
                  uint32[::1] col_indices not None):
        """
        :param num_examples:    The total number of examples
        :param num_labels:      The total number of labels
        :param row_indices:     An array of type `uint32`, shape `(num_examples + 1)`, that stores the indices of the
                                first element in `col_indices` that corresponds to a certain example. The index at the
                                last position is equal to `num_non_zero_values`
        :param col_indices:     An array of type `uint32`, shape `(num_non_zero_values)`, that stores the
                                column-indices, the relevant labels correspond to
        """
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.label_matrix_ptr = createCsrLabelMatrix(num_examples, num_labels, &row_indices[0], &col_indices[0])

    cdef ILabelMatrix* get_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()

    cdef IRowWiseLabelMatrix* get_row_wise_label_matrix_ptr(self):
        return self.label_matrix_ptr.get()
