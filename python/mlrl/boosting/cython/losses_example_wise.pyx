"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class ExampleWiseLoss(EvaluationMeasure):
    """
    A wrapper for the pure virtual C++ class `IExampleWiseLoss`.
    """

    cdef unique_ptr[IEvaluationMeasure] get_evaluation_measure_ptr(self):
        return <unique_ptr[IEvaluationMeasure]>move(self.loss_function_ptr)

    cdef unique_ptr[ISimilarityMeasure] get_similarity_measure_ptr(self):
        return <unique_ptr[ISimilarityMeasure]>move(self.loss_function_ptr)


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    """
    A wrapper for the C++ class `ExampleWiseLogisticLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <unique_ptr[IExampleWiseLoss]>make_unique[ExampleWiseLogisticLossImpl]()

    def __reduce__(self):
        return (ExampleWiseLogisticLoss, ())
