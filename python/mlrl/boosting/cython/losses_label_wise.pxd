from mlrl.common.cython._measures cimport IEvaluationMeasure, ISimilarityMeasure
from mlrl.common.cython.measures cimport EvaluationMeasure

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/losses/loss_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseLoss(IEvaluationMeasure, ISimilarityMeasure):
        pass


cdef extern from "boosting/losses/loss_label_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseLogisticLossImpl"boosting::LabelWiseLogisticLoss"(ILabelWiseLoss):
        pass


cdef extern from "boosting/losses/loss_label_wise_squared_error.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseSquaredErrorLossImpl"boosting::LabelWiseSquaredErrorLoss"(ILabelWiseLoss):
        pass


cdef extern from "boosting/losses/loss_label_wise_squared_hinge.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseSquaredHingeLossImpl"boosting::LabelWiseSquaredHingeLoss"(ILabelWiseLoss):
        pass


cdef class LabelWiseLoss(EvaluationMeasure):

    # Attributes:

    cdef unique_ptr[ILabelWiseLoss] loss_function_ptr


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    pass


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    pass


cdef class LabelWiseSquaredHingeLoss(LabelWiseLoss):
    pass
