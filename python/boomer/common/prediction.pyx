"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for making predictions using rule-based models.
"""
from boomer.common._arrays cimport uint8, c_matrix_uint8

import numpy as np


cdef class Predictor:
    """
    A base class for all classes that allow to make predictions based on rule-based models.
    """

    cpdef object predict(self, float32[:, ::1] x, uint32 num_labels, RuleModel model):
        """
        Obtains and returns the predictions for given examples.

        The feature matrix must be given as a dense C-contiguous array.

        :param x:           An array of type `float32`, shape `(num_examples, num_features)`, representing the features
                            of the examples to predict for
        :param num_labels:  The total number of labels
        :param model:       The model to be used for making predictions
        :return:            A `np.ndarray` or a `scipy.sparse.matrix`, shape `(num_examples, num_labels)`, representing
                            the predictions for individual examples and labels
        """
        pass

    cpdef object predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                             uint32 num_features, uint32 num_labels, RuleModel model):
        """
        Obtains and returns the predictions for given examples.

        The feature matrix must be given in compressed sparse row (CSR) format.

        :param x_data:          An array of type `float32`, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the training examples
        :param x_row_indices:   An array of type `uint32`, shape `(num_examples + 1)`, representing the indices of the
                                first element in `x_data` and `x_col_indices` that corresponds to a certain examples.
                                The index at the last position is equal to `num_non_zero_feature_values`
        :param x_col_indices:   An array of type `uint32`, shape `(num_non_zero_feature_values)`, representing the
                                column-indices of the examples, the values in `x_data` correspond to
        :param num_features:    The total number of features
        :param num_labels:      The total number of labels
        :param model:           The model to be used for making predictions
        :return:                A `np.ndarray` or a `scipy.sparse.matrix`, shape `(num_examples, num_labels)`,
                                representing the predictions for individual examples and labels
        """
        pass


cdef class DensePredictor(Predictor):
    """
    Allows to make predictions based on rule-based models that are stored in dense matrices.
    """

    def __cinit__(self, TransformationFunction transformation_function = None):
        """
        :param transformation_function: An (optional) transformation function to be applied to the raw predictions or
                                        None, if no transformation function should be applied
        """
        self.transformation_function = transformation_function

    cpdef object predict(self, float32[:, ::1] x, uint32 num_labels, RuleModel model):
        cdef float64[:, ::1] predictions = model.predict(x, num_labels)
        cdef TransformationFunction transformation_function = self.transformation_function

        if transformation_function is not None:
            return transformation_function.transform_matrix(predictions)
        else:
            return np.asarray(predictions)

    cpdef object predict_csr(self, float32[::1] x_data, uint32[::1] x_row_indices, uint32[::1] x_col_indices,
                             uint32 num_features, uint32 num_labels, RuleModel model):
        cdef float64[:, ::1] predictions = model.predict_csr(x_data, x_row_indices, x_col_indices, num_features,
                                                             num_labels)
        cdef TransformationFunction transformation_function = self.transformation_function

        if transformation_function is not None:
            return transformation_function.transform_matrix(predictions)
        else:
            return np.asarray(predictions)


cdef class TransformationFunction:
    """
    A base class for all transformation functions that may be applied to predictions.
    """

    cdef object transform_matrix(self, float64[:, ::1] m):
        """
        Applies the transformation function to a matrix.

        :param m:   An array of type `float64`, shape `(num_rows, num_cols)`, the transformation function should be
                    applied to
        :return:    A `np.ndarray` or `scipy.sparse.matrix`, shape `(num_rows, num_cols)`, representing the result of
                    the transformation
        """
        pass


cdef class SignFunction(TransformationFunction):
    """
    Transforms predictions according to the sign function (1 if x > 0, 0 otherwise).
    """

    cdef object transform_matrix(self, float64[:, ::1] m):
        cdef uint32 num_rows = m.shape[0]
        cdef uint32 num_cols = m.shape[1]
        cdef uint8[:, ::1] result = c_matrix_uint8(num_rows, num_cols)
        cdef uint32 r, c

        for r in range(num_rows):
            for c in range(num_cols):
                result[r, c] = m[r, c] > 0

        return np.asarray(result)
