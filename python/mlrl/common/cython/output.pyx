"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython._arrays cimport c_matrix_uint8, c_matrix_float64
from mlrl.common.cython._data cimport CContiguousView
from mlrl.common.cython.input cimport CContiguousFeatureMatrixImpl, CsrFeatureMatrixImpl

from libcpp.memory cimport shared_ptr, make_unique

from cython.operator cimport dereference

import numpy as np


cdef class Predictor:
    """
    A wrapper for the pure virtual C++ class `IPredictor`.
    """

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model):
        """
        Obtains and returns the predictions for given examples in a feature matrix that uses a C-contiguous array.

        :param feature_matrix:  A `CContiguousFeatureMatrix` that stores the examples to predict for
        :param model:           The `RuleModel` to be used for making predictions
        :return:                A `np.ndarray` or a `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that
                                stores the predictions for individual examples and labels
        """
        pass

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model):
        """
        Obtains and returns the predictions for given examples in a feature matrix that uses the compressed sparse row
        (CSR) format.

        :param feature_matrix:  A `CsrFeatureMatrix` that stores the examples to predict for
        :param model:           The `RuleModel` to be used for making predictions
        :return:                A `np.ndarray` or a `scipy.sparse`, shape `(num_examples, num_labels)`, that stores the
                                predictions for individual examples and labels
        """
        pass


cdef class AbstractNumericalPredictor(Predictor):
    """
    A base class for all classes that allow to predict numerical scores for given query examples.
    """

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model):
        cdef shared_ptr[CContiguousFeatureMatrixImpl] feature_matrix_ptr = feature_matrix.feature_matrix_ptr
        cdef uint32 num_examples = feature_matrix_ptr.get().getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[float64]] view_ptr = make_unique[CContiguousView[float64]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        self.predictor_ptr.get().predict(dereference(feature_matrix_ptr.get()), dereference(view_ptr.get()),
                                         dereference(model.model_ptr.get()))
        return np.asarray(prediction_matrix)

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model):
        cdef shared_ptr[CsrFeatureMatrixImpl] feature_matrix_ptr = feature_matrix.feature_matrix_ptr
        cdef uint32 num_examples = feature_matrix_ptr.get().getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[float64]] view_ptr = make_unique[CContiguousView[float64]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        self.predictor_ptr.get().predict(dereference(feature_matrix_ptr.get()), dereference(view_ptr.get()),
                                         dereference(model.model_ptr.get()))
        return np.asarray(prediction_matrix)


cdef class AbstractBinaryPredictor(Predictor):
    """
    A base class for all classes that allow to predict binary values for given query examples.
    """

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model):
        cdef shared_ptr[CContiguousFeatureMatrixImpl] feature_matrix_ptr = feature_matrix.feature_matrix_ptr
        cdef uint32 num_examples = feature_matrix_ptr.get().getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[uint8]] view_ptr = make_unique[CContiguousView[uint8]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        self.predictor_ptr.get().predict(dereference(feature_matrix_ptr.get()), dereference(view_ptr.get()),
                                         dereference(model.model_ptr.get()))
        return np.asarray(prediction_matrix)

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model):
        cdef shared_ptr[CsrFeatureMatrixImpl] feature_matrix_ptr = feature_matrix.feature_matrix_ptr
        cdef uint32 num_examples = feature_matrix_ptr.get().getNumRows()
        cdef uint32 num_labels = self.num_labels
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(num_examples, num_labels)
        cdef unique_ptr[CContiguousView[uint8]] view_ptr = make_unique[CContiguousView[uint8]](
            num_examples, num_labels, &prediction_matrix[0, 0])
        self.predictor_ptr.get().predict(dereference(feature_matrix_ptr.get()), dereference(view_ptr.get()),
                                         dereference(model.model_ptr.get()))
        return np.asarray(prediction_matrix)
