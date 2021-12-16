"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class LabelWiseLoss(EvaluationMeasure):
    """
    A wrapper for the pure virtual C++ class `ILabelWiseLoss`.
    """

    cdef unique_ptr[IEvaluationMeasure] get_evaluation_measure_ptr(self):
        return <unique_ptr[IEvaluationMeasure]>move(self.loss_function_ptr)

    cdef unique_ptr[ISimilarityMeasure] get_similarity_measure_ptr(self):
        return <unique_ptr[ISimilarityMeasure]>move(self.loss_function_ptr)


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseLogisticLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <unique_ptr[ILabelWiseLoss]>make_unique[LabelWiseLogisticLossImpl]()

    def __reduce__(self):
        return (LabelWiseLogisticLoss, ())


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseSquaredErrorLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <unique_ptr[ILabelWiseLoss]>make_unique[LabelWiseSquaredErrorLossImpl]()

    def __reduce__(self):
        return (LabelWiseSquaredErrorLoss, ())


cdef class LabelWiseSquaredHingeLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseSquaredHingeLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <unique_ptr[ILabelWiseLoss]>make_unique[LabelWiseSquaredHingeLossImpl]()

    def __reduce__(self):
        return (LabelWiseSquaredHingeLoss, ())
