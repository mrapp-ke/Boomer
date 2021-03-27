from libcpp.memory cimport shared_ptr


cdef extern from "common/thresholds/thresholds_factory.hpp" nogil:

    cdef cppclass IThresholdsFactory:
        pass


cdef class ThresholdsFactory:

    # Attributes:

    cdef shared_ptr[IThresholdsFactory] thresholds_factory_ptr
