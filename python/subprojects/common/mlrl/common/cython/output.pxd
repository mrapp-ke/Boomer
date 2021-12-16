from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython._data cimport CContiguousView
from mlrl.common.cython.input cimport CContiguousFeatureMatrixImpl, CsrFeatureMatrixImpl, LabelVectorSetImpl
from mlrl.common.cython.model cimport RuleModelImpl

from libcpp.memory cimport unique_ptr
from libcpp.forward_list cimport forward_list


cdef extern from "common/output/prediction_matrix_sparse_binary.hpp" nogil:

    cdef cppclass BinarySparsePredictionMatrix:

        ctypedef forward_list[uint32].const_iterator const_iterator

        # Functions:

        const_iterator row_cbegin(uint32 row)

        const_iterator row_cend(uint32 row)

        uint32 getNumRows()

        uint32 getNumCols()

        uint32 getNumNonZeroElements()


cdef extern from "common/output/predictor.hpp" nogil:

    cdef cppclass IPredictor[T]:

        # Functions:

        void predict(const CContiguousFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model, const LabelVectorSetImpl* labelVectors)

        void predict(const CsrFeatureMatrixImpl& featureMatrix, CContiguousView[T]& predictionMatrix,
                     const RuleModelImpl& model, const LabelVectorSetImpl* labelVectors)


cdef extern from "common/output/predictor_sparse.hpp" nogil:

    cdef cppclass ISparsePredictor[T](IPredictor[T]):

        # Functions:

        unique_ptr[BinarySparsePredictionMatrix] predictSparse(const CContiguousFeatureMatrixImpl& featureMatrix,
                                                               uint32 numLabels, const RuleModelImpl& model,
                                                               const LabelVectorSetImpl* labelVectors)

        unique_ptr[BinarySparsePredictionMatrix] predictSparse(const CsrFeatureMatrixImpl& featureMatrix,
                                                               uint32 numLabels, const RuleModelImpl& model,
                                                               const LabelVectorSetImpl* labelVectors)


cdef class Predictor:
    pass


cdef class SparsePredictor(Predictor):
    pass


cdef class AbstractNumericalPredictor(Predictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[IPredictor[float64]] predictor_ptr


cdef class AbstractBinaryPredictor(SparsePredictor):

    # Attributes:

    cdef uint32 num_labels

    cdef unique_ptr[ISparsePredictor[uint8]] predictor_ptr
