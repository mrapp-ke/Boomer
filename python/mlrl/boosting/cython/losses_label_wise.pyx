"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class LabelWiseLoss(EvaluationMeasure):
    """
    A wrapper for the pure virtual C++ class `ILabelWiseLoss`.
    """

    cdef shared_ptr[IEvaluationMeasure] get_evaluation_measure_ptr(self):
        return <shared_ptr[IEvaluationMeasure]>self.loss_function_ptr

    cdef shared_ptr[ISimilarityMeasure] get_similarity_measure_ptr(self):
        return <shared_ptr[ISimilarityMeasure]>self.loss_function_ptr


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseLogisticLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[ILabelWiseLoss]>make_shared[LabelWiseLogisticLossImpl]()

    def __reduce__(self):
        return (LabelWiseLogisticLoss, ())


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseSquaredErrorLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[ILabelWiseLoss]>make_shared[LabelWiseSquaredErrorLossImpl]()

    def __reduce__(self):
        return (LabelWiseSquaredErrorLoss, ())


cdef class LabelWiseSquaredHingeLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseSquaredHingeLoss`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[ILabelWiseLoss]>make_shared[LabelWiseSquaredHingeLossImpl]()

    def __reduce__(self):
        return (LabelWiseSquaredHingeLoss, ())
