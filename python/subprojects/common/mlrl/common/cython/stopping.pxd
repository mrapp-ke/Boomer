from mlrl.common.cython._types cimport uint32, float64

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/stopping/stopping_criterion.hpp" nogil:

    cdef cppclass IStoppingCriterion:
        pass


cdef extern from "common/stopping/stopping_criterion_size.hpp" nogil:

    cdef cppclass SizeStoppingCriterionImpl"SizeStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        SizeStoppingCriterionImpl(uint32 maxRules) except +


cdef extern from "common/stopping/stopping_criterion_time.hpp" nogil:

    cdef cppclass TimeStoppingCriterionImpl"TimeStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        TimeStoppingCriterionImpl(uint32 timeLimit) except +


cdef extern from "common/stopping/stopping_criterion_measure.hpp" nogil:

    cdef cppclass IAggregationFunction:
        pass


    cdef cppclass MinFunctionImpl"MinFunction"(IAggregationFunction):
        pass


    cdef cppclass MaxFunctionImpl"MaxFunction"(IAggregationFunction):
        pass


    cdef cppclass ArithmeticMeanFunctionImpl"ArithmeticMeanFunction"(IAggregationFunction):
        pass


    cdef cppclass MeasureStoppingCriterionImpl"MeasureStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        MeasureStoppingCriterionImpl(unique_ptr[IAggregationFunction] aggregationFunctionPtr, uint32 minRules,
                                     uint32 updateInterval, uint32 stopInterval, uint32 numPast, uint32 numRecent,
                                     float64 minImprovement, bool forceStop) except +


cdef class StoppingCriterion:

    # Attributes:

    cdef unique_ptr[IStoppingCriterion] stopping_criterion_ptr


cdef class SizeStoppingCriterion(StoppingCriterion):
    pass


cdef class TimeStoppingCriterion(StoppingCriterion):
    pass

cdef class AggregationFunction:

    # Attributes:

    cdef unique_ptr[IAggregationFunction] aggregation_function_ptr


cdef class MinFunction(AggregationFunction):
    pass


cdef class MaxFunction(AggregationFunction):
    pass


cdef class ArithmeticMeanFunction(AggregationFunction):
    pass


cdef class MeasureStoppingCriterion(StoppingCriterion):
    pass
