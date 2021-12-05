"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython._arrays cimport array_uint8, array_uint32, c_matrix_uint8, c_matrix_float64
from mlrl.common.cython._data cimport CContiguousView
from mlrl.common.cython.input cimport CContiguousFeatureMatrix, CContiguousFeatureMatrixImpl, CsrFeatureMatrixImpl, \
    CsrFeatureMatrix, LabelVectorSet
from mlrl.common.cython.model cimport RuleModel

from libcpp.memory cimport make_unique

from cython.operator cimport dereference, postincrement

from scipy.sparse import csr_matrix
import numpy as np


cdef class Predictor:
    """
    A wrapper for the pure virtual C++ class `IPredictor`.
    """

    def predict_dense(self, CContiguousFeatureMatrix feature_matrix not None, RuleModel model not None,
                LabelVectorSet label_vectors) -> np.ndarray:
        """
        Obtains and returns dense predictions for given examples in a feature matrix that uses a C-contiguous array.

        :param feature_matrix:  A `CContiguousFeatureMatrix` that stores the examples to predict for
        :param model:           The `RuleModel` to be used for making predictions
        :param label_vectors    A `LabelVectorSet` that stores all known label vectors or None, if no such set is
                                available
        :return:                A `np.ndarray`, shape `(num_examples, num_labels)`, that stores the predictions for
                                individual examples and labels
        """
        pass

    def predict_dense_csr(self, CsrFeatureMatrix feature_matrix not None, RuleModel model not None,
                          LabelVectorSet label_vectors) -> np.ndarray:
        """
        Obtains and returns dense predictions for given examples in a feature matrix that uses the compressed sparse row
        (CSR) format.

        :param feature_matrix:  A `CsrFeatureMatrix` that stores the examples to predict for
        :param model:           The `RuleModel` to be used for making predictions
        :param label_vectors    A `LabelVectorSet` that stores all known label vectors or None, if no such set is
                                available
        :return:                A `np.ndarray`, shape `(num_examples, num_labels)`, that stores the predictions for
                                individual examples and labels
        """
        pass


cdef class SparsePredictor(Predictor):
    """
    A wrapper for the pure virtual C++ class `ISparsePredictor`.
    """

    def predict_sparse(self, CContiguousFeatureMatrix feature_matrix not None, RuleModel model not None,
                       LabelVectorSet label_vectors) -> csr_matrix:
        """
        Obtains and returns sparse predictions for given examples in a feature matrix that uses a C-contiguous array.

        :param feature_matrix:  A `CContiguousFeatureMatrix` that stores the examples to predict for
        :param model:           The `RuleModel` to be used for making predictions
        :param label_vectors    A `LabelVectorSet` that stores all known label vectors or None, if no such set is
                                available
        :return:                A `scipy.sparse.csr_matrix`, shape `(num_examples, num_labels)`, that stores the
                                predictions for individual examples and labels
        """
        pass

    def predict_sparse_csr(self, CsrFeatureMatrix feature_matrix not None, RuleModel model not None,
                           LabelVectorSet label_vectors) -> csr_matrix:
        """
        Obtains and returns dense predictions for given examples in a feature matrix that uses the compressed sparse row
        (CSR) format.

        :param feature_matrix:  A `CsrFeatureMatrix` that stores the examples to predict for
        :param model:           The `RuleModel` to be used for making predictions
        :param label_vectors    A `LabelVectorSet` that stores all known label vectors or None, if no such set is
                                available
        :return:                A `scipy.sparse.csr_matrix`, shape `(num_examples, num_labels)`, that stores the
                                predictions for individual examples and labels
        """
        pass


cdef class AbstractNumericalPredictor(Predictor):
    """
    A base class for all classes that allow to predict numerical scores for given query examples.
    """

    def predict(self, CContiguousFeatureMatrix feature_matrix not None, RuleModel model not None,
                LabelVectorSet label_vectors):
        cdef CContiguousFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_examples = feature_matrix_ptr.getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[float64]] view_ptr = make_unique[CContiguousView[float64]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        cdef LabelVectorSetImpl* label_vectors_ptr = <LabelVectorSetImpl*>NULL if label_vectors is None \
                                                        else label_vectors.label_vector_set_ptr.get()
        self.predictor_ptr.get().predict(dereference(feature_matrix_ptr), dereference(view_ptr),
                                         dereference(model.model_ptr), label_vectors_ptr)
        return np.asarray(prediction_matrix)

    def predict_csr(self, CsrFeatureMatrix feature_matrix not None, RuleModel model not None,
                    LabelVectorSet label_vectors):
        cdef CsrFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_examples = feature_matrix_ptr.getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[float64]] view_ptr = make_unique[CContiguousView[float64]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        cdef LabelVectorSetImpl* label_vectors_ptr = <LabelVectorSetImpl*>NULL if label_vectors is None \
                                                        else label_vectors.label_vector_set_ptr.get()
        self.predictor_ptr.get().predict(dereference(feature_matrix_ptr), dereference(view_ptr),
                                         dereference(model.model_ptr), label_vectors_ptr)
        return np.asarray(prediction_matrix)


cdef inline object __create_csr_matrix(BinarySparsePredictionMatrix* prediction_matrix):
    cdef uint32 num_rows = prediction_matrix.getNumRows()
    cdef uint32 num_cols = prediction_matrix.getNumCols()
    cdef uint32 num_non_zero_elements = prediction_matrix.getNumNonZeroElements()
    cdef uint8[::1] data = array_uint8(num_non_zero_elements) if num_non_zero_elements > 0 else None
    cdef uint32[::1] col_indices = array_uint32(num_non_zero_elements) if num_non_zero_elements > 0 else None
    cdef uint32[::1] row_indices = array_uint32(num_rows + 1)
    cdef BinarySparsePredictionMatrix.const_iterator it
    cdef BinarySparsePredictionMatrix.const_iterator end
    cdef uint32 row_index
    cdef uint32 i = 0

    for row_index in range(num_rows):
        it = prediction_matrix.row_cbegin(row_index)
        end = prediction_matrix.row_cend(row_index)
        row_indices[row_index] = i

        while it != end:
            col_indices[i] = dereference(it)
            data[i] = 1
            i += 1
            postincrement(it)

    row_indices[num_rows] = i
    return csr_matrix((np.asarray([] if data is None else data), np.asarray([] if col_indices is None else col_indices),
                       np.asarray(row_indices)), shape=(num_rows, num_cols))


cdef class AbstractBinaryPredictor(SparsePredictor):
    """
    A base class for all classes that allow to predict binary values for given query examples.
    """

    def predict_dense(self, CContiguousFeatureMatrix feature_matrix not None, RuleModel model not None,
                      LabelVectorSet label_vectors):
        cdef CContiguousFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_examples = feature_matrix_ptr.getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[uint8]] view_ptr = make_unique[CContiguousView[uint8]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        cdef LabelVectorSetImpl* label_vectors_ptr = <LabelVectorSetImpl*>NULL if label_vectors is None \
                                                        else label_vectors.label_vector_set_ptr.get()
        cdef IPredictor[uint8]* predictor_ptr = self.predictor_ptr.get()
        predictor_ptr.predict(dereference(feature_matrix_ptr), dereference(view_ptr), dereference(model.model_ptr),
                              label_vectors_ptr)
        return np.asarray(prediction_matrix)

    def predict_dense_csr(self, CsrFeatureMatrix feature_matrix not None, RuleModel model not None,
                          LabelVectorSet label_vectors):
        cdef CsrFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_examples = feature_matrix_ptr.getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[uint8]] view_ptr = make_unique[CContiguousView[uint8]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        cdef LabelVectorSetImpl* label_vectors_ptr = <LabelVectorSetImpl*>NULL if label_vectors is None \
                                                        else label_vectors.label_vector_set_ptr.get()
        cdef IPredictor[uint8]* predictor_ptr = self.predictor_ptr.get()
        predictor_ptr.predict(dereference(feature_matrix_ptr), dereference(view_ptr), dereference(model.model_ptr),
                              label_vectors_ptr)
        return np.asarray(prediction_matrix)

    def predict_sparse(self, CContiguousFeatureMatrix feature_matrix not None, RuleModel model not None,
                       LabelVectorSet label_vectors) -> csr_matrix:
        cdef CContiguousFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_labels = self.num_labels
        cdef LabelVectorSetImpl* label_vectors_ptr = <LabelVectorSetImpl*>NULL if label_vectors is None \
                                                        else label_vectors.label_vector_set_ptr.get()
        cdef ISparsePredictor[uint8]* predictor_ptr = self.predictor_ptr.get()
        cdef unique_ptr[BinarySparsePredictionMatrix] prediction_matrix_ptr = predictor_ptr.predict(
            dereference(feature_matrix_ptr), num_labels, dereference(model.model_ptr), label_vectors_ptr)
        return __create_csr_matrix(prediction_matrix_ptr.get())

    def predict_sparse_csr(self, CsrFeatureMatrix feature_matrix not None, RuleModel model not None,
                           LabelVectorSet label_vectors) -> csr_matrix:
        cdef CsrFeatureMatrixImpl* feature_matrix_ptr = feature_matrix.feature_matrix_ptr.get()
        cdef uint32 num_labels = self.num_labels
        cdef LabelVectorSetImpl* label_vectors_ptr = <LabelVectorSetImpl*>NULL if label_vectors is None \
                                                        else label_vectors.label_vector_set_ptr.get()
        cdef ISparsePredictor[uint8]* predictor_ptr = self.predictor_ptr.get()
        cdef unique_ptr[BinarySparsePredictionMatrix] prediction_matrix_ptr = predictor_ptr.predict(
            dereference(feature_matrix_ptr), num_labels, dereference(model.model_ptr), label_vectors_ptr)
        return __create_csr_matrix(prediction_matrix_ptr.get())
