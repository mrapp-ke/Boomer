"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different differentiable loss functions.
"""
from libcpp.memory cimport make_shared


cdef class ExampleWiseLoss:
    """
    A wrapper for the abstract C++ class `AbstractExampleWiseLoss`.
    """
    pass


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    """
    A wrapper for the C++ class `ExampleWiseLogisticLossImpl`.
    """

    def __cinit__(self):
        self.loss_function_ptr = <shared_ptr[AbstractExampleWiseLoss]>make_shared[ExampleWiseLogisticLossImpl]()
