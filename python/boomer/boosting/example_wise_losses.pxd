from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_losses.h" namespace "boosting" nogil:

    cdef cppclass AbstractExampleWiseLoss:
        pass


    cdef cppclass ExampleWiseLogisticLossImpl(AbstractExampleWiseLoss):
        pass


cdef class ExampleWiseLoss:

    # Attributes:

    cdef shared_ptr[AbstractExampleWiseLoss] loss_function_ptr


cdef class ExampleWiseLogisticLoss(ExampleWiseLoss):
    pass
