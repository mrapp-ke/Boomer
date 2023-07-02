from libcpp cimport bool

from mlrl.common.cython._types cimport float64, uint8, uint32


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


cdef extern from "common/stopping/aggregation_function.hpp" nogil:

    cpdef enum AggregationFunctionImpl"AggregationFunction":

        MIN"AggregationFunction::MIN" = 0

        MAX"AggregationFunction::MAX" = 1

        ARITHMETIC_MEAN"AggregationFunction::ARITHMETIC_MEAN" = 2


cdef extern from "common/stopping/global_pruning_pre.hpp" nogil:

    cdef cppclass IPrePruningConfig:

        # Functions:

        AggregationFunctionImpl getAggregationFunction() const

        IPrePruningConfig& setAggregationFunction(AggregationFunctionImpl aggregationFunction) except +

        bool isHoldoutSetUsed() const

        IPrePruningConfig& setUseHoldoutSet(bool useHoldoutSet) except +

        bool isRemoveUnusedRules() const

        IPrePruningConfig& setRemoveUnusedRules(bool removeUnusedRules) except +

        uint32 getMinRules() const

        IPrePruningConfig& setMinRules(uint32 minRules) except +

        uint32 getUpdateInterval() const

        IPrePruningConfig& setUpdateInterval(uint32 updateInterval) except +

        uint32 getStopInterval() const;

        IPrePruningConfig& setStopInterval(uint32 stopInterval) except +

        uint32 getNumPast() const

        IPrePruningConfig& setNumPast(uint32 numPast) except +

        uint32 getNumCurrent() const

        IPrePruningConfig& setNumCurrent(uint32 numCurrent) except +

        float64 getMinImprovement() const

        IPrePruningConfig& setMinImprovement(float64 minImprovement) except +


cdef extern from "common/stopping/global_pruning_post.hpp" nogil:

    cdef cppclass IPostPruningConfig:

        # Functions:

        bool isHoldoutSetUsed() const

        IPostPruningConfig& setUseHoldoutSet(bool useHoldoutSet) except +

        bool isRemoveUnusedRules() const

        IPrePruningConfig& setRemoveUnusedRules(bool removeUnusedRules) except +

        uint32 getMinRules() const

        IPostPruningConfig& setMinRules(uint32 minRules) except +

        uint32 getInterval() const

        IPostPruningConfig& setInterval(uint32 interval) except +


cdef class SizeStoppingCriterionConfig:

    # Attributes:

    cdef ISizeStoppingCriterionConfig* config_ptr


cdef class TimeStoppingCriterionConfig:

    # Attributes:

    cdef ITimeStoppingCriterionConfig* config_ptr


cdef class PrePruningConfig:

    # Attributes:

    cdef IPrePruningConfig* config_ptr


cdef class PostPruningConfig:

    # Attributes:

    cdef IPostPruningConfig* config_ptr
