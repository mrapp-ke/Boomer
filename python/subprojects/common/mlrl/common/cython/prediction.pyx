"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal

from libcpp.utility cimport move

from mlrl.common.cython._arrays cimport array_uint32, c_matrix_float64, c_matrix_uint8, c_view_float64, c_view_uint8, \
    view_uint32

import numpy as np

from scipy.sparse import csr_matrix


cdef class IncrementalBinaryPredictor:
    """
    Allows to predict binary labels for given query examples incrementally.
    """

    def has_next(self) -> bool:
        """
        Returns whether there are any remaining ensemble members that have not been used yet or not.

        :return: True, if there are any remaining ensemble members, False otherwise
        """
        return self.predictor_ptr.get().hasNext()

    def get_num_next(self) -> int:
        """
        Returns the number of remaining ensemble members that have not been used yet.

        :return: The number of remaining ensemble members
        """
        return self.predictor_ptr.get().getNumNext()

    def apply_next(self, uint32 step_size) -> np.ndarray:
        """
        Updates the current predictions by considering several of the remaining ensemble members. If not enough ensemble
        members are remaining, only the available ones will be used for updating the current predictions.

        :param step_size:   The number of additional ensemble members to be considered for prediction
        :return:            A `numpy.ndarray` of type `uint8`, shape `(num_examples, num_labels)`, that stores the
                            updated predictions
        """
        assert_greater_or_equal('step_size', step_size, 1)
        cdef DensePredictionMatrix[uint8]* prediction_matrix_ptr = &self.predictor_ptr.get().applyNext(step_size)
        cdef uint32 num_rows = prediction_matrix_ptr.getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.getNumCols()
        cdef uint8* array = prediction_matrix_ptr.get()
        cdef uint8[:, ::1] prediction_matrix = c_view_uint8(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)


cdef class BinaryPredictor:
    """
    Allows to predict binary labels for given query examples.
    """

    def predict(self, uint32 max_rules) -> np.ndarray:
        """
        Obtains and returns predictions for all query examples.

        :param max_rules    The maximum number of rules to be used for prediction or 0, if the number of rules should
                            not be restricted
        :return:            A `numpy.ndarray` of type `uint8`, shape `(num_examples, num_labels)`, that stores the
                            predictions
        """
        cdef unique_ptr[DensePredictionMatrix[uint8]] prediction_matrix_ptr = \
            self.predictor_ptr.get().predict(max_rules)
        cdef uint32 num_rows = prediction_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.get().getNumCols()
        cdef uint8* array = prediction_matrix_ptr.get().release()
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)

    def can_predict_incrementally(self) -> bool:
        """
        Returns whether the predictor allows to obtain predictions incrementally or not.

        :return: True, if the predictor allows to obtain predictions incrementally, False otherwise
        """
        return self.predictor_ptr.get().canPredictIncrementally()

    def create_incremental_predictor(self, uint32 max_rules) -> IncrementalBinaryPredictor:
        """
        Creates and returns a predictor that allows to predict binary labels incrementally. If incremental prediction is
        not supported, a `RuntimeError` is thrown.

        :param max_rules:   The maximum number of rules to be used for prediction. Must be at least 1 or 0, if the
                            number of rules should not be restricted
        :return:            A predictor that allows to predict binary labels incrementally
        """
        if max_rules != 0:
            assert_greater_or_equal('max_rules', max_rules, 1)
        cdef IncrementalBinaryPredictor predictor = IncrementalBinaryPredictor.__new__(IncrementalBinaryPredictor)
        predictor.predictor_ptr = move(self.predictor_ptr.get().createIncrementalPredictor(max_rules))
        return predictor


cdef class IncrementalSparseBinaryPredictor:
    """
    Allows to predict sparse binary labels for given query examples incrementally.
    """

    def has_next(self) -> bool:
        """
        Returns whether there are any remaining ensemble members that have not been used yet or not.

        :return: True, if there are any remaining ensemble members, False otherwise
        """
        return self.predictor_ptr.get().hasNext()

    def get_num_next(self) -> int:
        """
        Returns the number of remaining ensemble members that have not been used yet.

        :return: The number of remaining ensemble members
        """
        return self.predictor_ptr.get().getNumNext()

    def apply_next(self, uint32 step_size) -> np.ndarray:
        """
        Updates the current predictions by considering several of the remaining ensemble members. If not enough ensemble
        members are remaining, only the available ones will be used for updating the current predictions.

        :param step_size:   The number of additional ensemble members to be considered for prediction
        :return:            A `scipy.sparse.csr_matrix` of type `uint8`, shape `(num_examples, num_labels)` that stores
                            the predictions
        """
        cdef BinarySparsePredictionMatrix* prediction_matrix_ptr = &self.predictor_ptr.get().applyNext(step_size)
        cdef uint32 num_rows = prediction_matrix_ptr.getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.getNumCols()
        cdef uint32 num_non_zero_elements = prediction_matrix_ptr.getNumNonZeroElements()
        cdef uint32* row_indices = prediction_matrix_ptr.getRowIndices()
        cdef uint32* col_indices = prediction_matrix_ptr.getColIndices()
        data = np.ones(shape=(num_non_zero_elements), dtype=np.uint8) if num_non_zero_elements > 0 else np.asarray([])
        indices = np.asarray(view_uint32(col_indices, num_non_zero_elements) if num_non_zero_elements > 0 else [])
        indptr = np.asarray(view_uint32(row_indices, num_rows + 1))
        return csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))


cdef class SparseBinaryPredictor:
    """
    Allows to predict sparse binary labels for given query examples.
    """

    def predict(self, uint32 max_rules) -> csr_matrix:
        """
        Obtains and returns predictions for all query examples.

        :param max_rules:   The maximum number of rules to be used for prediction or 0, if the number of rules should
                            not be restricted
        :return:            A `scipy.sparse.csr_matrix` of type `uint8`, shape `(num_examples, num_labels)` that stores
                            the predictions
        """
        cdef unique_ptr[BinarySparsePredictionMatrix] prediction_matrix_ptr = \
            self.predictor_ptr.get().predict(max_rules)
        cdef uint32 num_rows = prediction_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.get().getNumCols()
        cdef uint32 num_non_zero_elements = prediction_matrix_ptr.get().getNumNonZeroElements()
        cdef uint32* row_indices = prediction_matrix_ptr.get().releaseRowIndices()
        cdef uint32* col_indices = prediction_matrix_ptr.get().releaseColIndices()
        data = np.ones(shape=(num_non_zero_elements), dtype=np.uint8) if num_non_zero_elements > 0 else np.asarray([])
        indices = np.asarray(array_uint32(col_indices, num_non_zero_elements) if num_non_zero_elements > 0 else [])
        indptr = np.asarray(array_uint32(row_indices, num_rows + 1))
        return csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))

    def can_predict_incrementally(self) -> bool:
        """
        Returns whether the predictor allows to obtain predictions incrementally or not.

        :return: True, if the predictor allows to obtain predictions incrementally, False otherwise
        """
        return self.predictor_ptr.get().canPredictIncrementally()

    def create_incremental_predictor(self, uint32 max_rules) -> IncrementalSparseBinaryPredictor:
        """
        Creates and returns a predictor that allows to predict sparse binary labels incrementally. If incremental
        prediction is not supported, a `RuntimeError` is thrown.

        :param max_rules:   The maximum number of rules to be used for prediction. Must be at least 1 or 0, if the
                            number of rules should not be restricted
        :return:            A predictor that allows to predict sparse binary labels incrementally
        """
        if max_rules != 0:
            assert_greater_or_equal('max_rules', max_rules, 1)
        cdef IncrementalSparseBinaryPredictor predictor = \
            IncrementalSparseBinaryPredictor.__new__(IncrementalSparseBinaryPredictor)
        predictor.predictor_ptr = move(self.predictor_ptr.get().createIncrementalPredictor(max_rules))
        return predictor


cdef class IncrementalScorePredictor:
    """
    Allows to predict regression scores for given query examples incrementally.
    """

    def has_next(self) -> bool:
        """
        Returns whether there are any remaining ensemble members that have not been used yet or not.

        :return: True, if there are any remaining ensemble members, False otherwise
        """
        return self.predictor_ptr.get().hasNext()

    def get_num_next(self) -> int:
        """
        Returns the number of remaining ensemble members that have not been used yet.

        :return: The number of remaining ensemble members
        """
        return self.predictor_ptr.get().getNumNext()

    def apply_next(self, uint32 step_size) -> np.ndarray:
        """
        Updates the current predictions by considering several of the remaining ensemble members. If not enough ensemble
        members are remaining, only the available ones will be used for updating the current predictions.

        :param step_size:   The number of additional ensemble members to be considered for prediction
        :return:            A `numpy.ndarray` of type `float64`, shape `(num_examples, num_labels)`, that stores the
                            updated predictions
        """
        cdef DensePredictionMatrix[float64]* prediction_matrix_ptr = &self.predictor_ptr.get().applyNext(step_size)
        cdef uint32 num_rows = prediction_matrix_ptr.getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.getNumCols()
        cdef float64* array = prediction_matrix_ptr.get()
        cdef float64[:, ::1] prediction_matrix = c_view_float64(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)

cdef class ScorePredictor:
    """
    Allows to predict regression scores for given query examples.
    """

    def predict(self, uint32 max_rules) -> np.ndarray:
        """
        Obtains and returns predictions for all query examples.

        :param max_rules:   The maximum number of rules to be used for prediction or 0, if the number of rules should
                            not be restricted
        :return:            A `numpy.ndarray` of type `float64`, shape `(num_examples, num_labels)`, that stores the
                            predictions
        """
        cdef unique_ptr[DensePredictionMatrix[float64]] prediction_matrix_ptr = \
            self.predictor_ptr.get().predict(max_rules)
        cdef uint32 num_rows = prediction_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.get().getNumCols()
        cdef float64* array = prediction_matrix_ptr.get().release()
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)

    def can_predict_incrementally(self) -> bool:
        """
        Returns whether the predictor allows to obtain predictions incrementally or not.

        :return: True, if the predictor allows to obtain predictions incrementally, False otherwise
        """
        return self.predictor_ptr.get().canPredictIncrementally()

    def create_incremental_predictor(self, uint32 max_rules) -> IncrementalScorePredictor:
        """
        Creates and returns a predictor that allows to predict regression scores incrementally. If incremental
        prediction is not supported, a `RuntimeError` is thrown.

        :param max_rules:   The maximum number of rules to be used for prediction. Must be at least 1 or 0, if the
                            number of rules should not be restricted
        :return:            A predictor that allows to predict regression scores incrementally
        """
        if max_rules != 0:
            assert_greater_or_equal('max_rules', max_rules, 1)
        cdef IncrementalScorePredictor predictor = IncrementalScorePredictor.__new__(IncrementalScorePredictor)
        predictor.predictor_ptr = move(self.predictor_ptr.get().createIncrementalPredictor(max_rules))
        return predictor


cdef class IncrementalProbabilityPredictor:
    """
    Allows to predict probability estimates for given query examples incrementally.
    """

    def has_next(self) -> bool:
        """
        Returns whether there are any remaining ensemble members that have not been used yet or not.

        :return: True, if there are any remaining ensemble members, False otherwise
        """
        return self.predictor_ptr.get().hasNext()

    def get_num_next(self) -> int:
        """
        Returns the number of remaining ensemble members that have not been used yet.

        :return: The number of remaining ensemble members
        """
        return self.predictor_ptr.get().getNumNext()

    def apply_next(self, uint32 step_size) -> np.ndarray:
        """
        Updates the current predictions by considering several of the remaining ensemble members. If not enough ensemble
        members are remaining, only the available ones will be used for updating the current predictions.

        :param step_size:   The number of additional ensemble members to be considered for prediction
        :return:            A `numpy.ndarray` of type `float64`, shape `(num_examples, num_labels)`, that stores the
                            updated predictions
        """
        cdef DensePredictionMatrix[float64]* prediction_matrix_ptr = &self.predictor_ptr.get().applyNext(step_size)
        cdef uint32 num_rows = prediction_matrix_ptr.getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.getNumCols()
        cdef float64* array = prediction_matrix_ptr.get()
        cdef float64[:, ::1] prediction_matrix = c_view_float64(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)


cdef class ProbabilityPredictor:
    """
    Allows to predict probability estimates for given query examples.
    """

    def predict(self, uint32 max_rules) -> np.ndarray:
        """
        Obtains and returns predictions for all query examples.

        :param max_rules:   The maximum number of rules to be used for prediction or 0, if the number of rules should
                            not be restricted
        :return:            A `numpy.ndarray` of type `float64`, shape `(num_examples, num_labels)`, that stores the
                            predictions
        """
        cdef unique_ptr[DensePredictionMatrix[float64]] prediction_matrix_ptr = \
            self.predictor_ptr.get().predict(max_rules)
        cdef uint32 num_rows = prediction_matrix_ptr.get().getNumRows()
        cdef uint32 num_cols = prediction_matrix_ptr.get().getNumCols()
        cdef float64* array = prediction_matrix_ptr.get().release()
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(array, num_rows, num_cols)
        return np.asarray(prediction_matrix)

    def can_predict_incrementally(self) -> bool:
        """
        Returns whether the predictor allows to obtain predictions incrementally or not.

        :return: True, if the predictor allows to obtain predictions incrementally, False otherwise
        """
        return self.predictor_ptr.get().canPredictIncrementally()

    def create_incremental_predictor(self, uint32 max_rules) -> IncrementalProbabilityPredictor:
        """
        Creates and returns a predictor that allows to predict probability estimates incrementally. If incremental
        prediction is not supported, a `RuntimeError` is thrown.

        :param max_rules:   The maximum number of rules to be used for prediction. Must be at least 1 or 0, if the
                            number of rules should not be restricted
        :return:            A predictor that allows to predict probability estimates incrementally
        """
        if max_rules != 0:
            assert_greater_or_equal('max_rules', max_rules, 1)
        cdef IncrementalProbabilityPredictor predictor = \
            IncrementalProbabilityPredictor.__new__(IncrementalProbabilityPredictor)
        predictor.predictor_ptr = move(self.predictor_ptr.get().createIncrementalPredictor(max_rules))
        return predictor
