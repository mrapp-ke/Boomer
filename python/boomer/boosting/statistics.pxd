from boomer.common.statistics cimport AbstractStatistics


cdef extern from "cpp/statistics.h" namespace "boosting" nogil:

    cdef cppclass AbstractGradientStatistics(AbstractStatistics):
        pass
