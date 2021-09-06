from libcpp.memory cimport unique_ptr


cdef extern from "common/thresholds/thresholds_factory.hpp" nogil:

    cdef cppclass IThresholdsFactory:
        pass


cdef class ThresholdsFactory:

    # Attributes:

    cdef unique_ptr[IThresholdsFactory] thresholds_factory_ptr
