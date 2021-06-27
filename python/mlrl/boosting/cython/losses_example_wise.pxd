from mlrl.common.cython._measures cimport IEvaluationMeasure, ISimilarityMeasure
from mlrl.common.cython.measures cimport EvaluationMeasure

from libcpp.memory cimport shared_ptr


cdef extern from "boosting/losses/loss_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseLoss(IEvaluationMeasure, ISimilarityMeasure):
        pass


cdef extern from "boosting/losses/loss_example_wise_logistic.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseLogisticLossImpl"boosting::ExampleWiseLogisticLoss"(IExampleWiseLoss):
        pass


cdef class ExampleWiseLoss(EvaluationMeasure):

    # Attributes:

    cdef shared_ptr[IExampleWiseLoss] loss_function_ptr


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    pass
