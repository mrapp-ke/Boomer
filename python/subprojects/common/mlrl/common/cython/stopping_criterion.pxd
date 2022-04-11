from mlrl.common.cython._types cimport uint8, uint32, float64

from libcpp cimport bool


cdef extern from "common/stopping/stopping_criterion_size.hpp" nogil:

    cdef cppclass ISizeStoppingCriterionConfig:

        # Functions:

        uint32 getMaxRules() const

        ISizeStoppingCriterionConfig& setMaxRules(uint32 maxRules) except +


cdef extern from "common/stopping/stopping_criterion_time.hpp" nogil:

    cdef cppclass ITimeStoppingCriterionConfig:

        # Functions:

        uint32 getTimeLimit() const

        ITimeStoppingCriterionConfig& setTimeLimit(uint32 timeLimit) except +


cdef extern from "common/stopping/stopping_criterion_measure.hpp" nogil:

    cpdef enum AggregationFunctionImpl"IMeasureStoppingCriterionConfig::AggregationFunction":

        MIN"IMeasureStoppingCriterionConfig::AggregationFunction::MIN" = 0

        MAX"IMeasureStoppingCriterionConfig::AggregationFunction::MAX" = 1

        ARITHMETIC_MEAN"IMeasureStoppingCriterionConfig::AggregationFunction::ARITHMETIC_MEAN" = 2


    cdef cppclass IMeasureStoppingCriterionConfig:

        # Functions:

        AggregationFunctionImpl getAggregationFunction() const

        IMeasureStoppingCriterionConfig& setAggregationFunction(AggregationFunctionImpl aggregationFunction) except +

        uint32 getMinRules() const

        IMeasureStoppingCriterionConfig& setMinRules(uint32 minRules) except +

        uint32 getUpdateInterval() const

        IMeasureStoppingCriterionConfig& setUpdateInterval(uint32 updateInterval) except +

        uint32 getStopInterval() const;

        IMeasureStoppingCriterionConfig& setStopInterval(uint32 stopInterval) except +

        uint32 getNumPast() const

        IMeasureStoppingCriterionConfig& setNumPast(uint32 numPast) except +

        uint32 getNumCurrent() const

        IMeasureStoppingCriterionConfig& setNumCurrent(uint32 numCurrent) except +

        float64 getMinImprovement() const

        IMeasureStoppingCriterionConfig& setMinImprovement(float64 minImprovement) except +

        bool getForceStop() const

        IMeasureStoppingCriterionConfig& setForceStop(bool forceStop) except +


cdef class SizeStoppingCriterionConfig:

    # Attributes:

    cdef ISizeStoppingCriterionConfig* config_ptr


cdef class TimeStoppingCriterionConfig:

    # Attributes:

    cdef ITimeStoppingCriterionConfig* config_ptr


cdef class MeasureStoppingCriterionConfig:


    # Attributes:

    cdef IMeasureStoppingCriterionConfig* config_ptr
