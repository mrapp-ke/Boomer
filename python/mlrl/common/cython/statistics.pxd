from libcpp.memory cimport shared_ptr


cdef extern from "common/statistics/statistics_provider_factory.hpp" nogil:

    cdef cppclass IStatisticsProviderFactory:
        pass


cdef class StatisticsProviderFactory:

    # Attributes:

    cdef shared_ptr[IStatisticsProviderFactory] statistics_provider_factory_ptr
