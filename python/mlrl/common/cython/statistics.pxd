from libcpp.memory cimport unique_ptr


cdef extern from "common/statistics/statistics_provider_factory.hpp" nogil:

    cdef cppclass IStatisticsProviderFactory:
        pass


cdef class StatisticsProviderFactory:

    # Attributes:

    cdef unique_ptr[IStatisticsProviderFactory] statistics_provider_factory_ptr
