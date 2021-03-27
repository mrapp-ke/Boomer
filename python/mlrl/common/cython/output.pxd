from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython._data cimport CContiguousView
from mlrl.common.cython.input cimport CContiguousFeatureMatrix, CContiguousFeatureMatrixImpl, CsrFeatureMatrix, \
    CsrFeatureMatrixImpl
from mlrl.common.cython.model cimport RuleModel, RuleModelImpl

from libcpp.memory cimport unique_ptr


cdef extern from "common/output/predictor.hpp" nogil:

    cdef cppclass IPredictor[T]:

        # Functions:

        void predict(const CContiguousFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model)

        void predict(const CsrFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model)


cdef class Predictor:

    # Functions:

    cpdef object predict(self, CContiguousFeatureMatrix feature_matrix, RuleModel model)

    cpdef object predict_csr(self, CsrFeatureMatrix feature_matrix, RuleModel model)


cdef class AbstractNumericalPredictor(Predictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IPredictor[float64]] predictor_ptr


cdef class AbstractBinaryPredictor(Predictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IPredictor[uint8]] predictor_ptr
