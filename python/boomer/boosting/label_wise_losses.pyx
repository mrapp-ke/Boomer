"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different differentiable loss functions.
"""
from libcpp.memory cimport make_shared


cdef class LabelWiseLoss:
    """
    A wrapper for the abstract C++ class `AbstractLabelWiseLoss`.
    """
    pass


cdef class LabelWiseLogisticLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseLogisticLossImpl`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[AbstractLabelWiseLoss]>make_shared[LabelWiseLogisticLossImpl]()


cdef class LabelWiseSquaredErrorLoss(LabelWiseLoss):
    """
    A wrapper for the C++ class `LabelWiseSquaredErrorLossImpl`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[AbstractLabelWiseLoss]>make_shared[LabelWiseSquaredErrorLossImpl]()
