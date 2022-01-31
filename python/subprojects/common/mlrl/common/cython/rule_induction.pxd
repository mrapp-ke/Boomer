from mlrl.common.cython._types cimport uint32

from libcpp cimport bool


cdef extern from "common/rule_induction/rule_induction_top_down.hpp" nogil:

    cdef cppclass ITopDownRuleInductionConfig:

        # Functions:

        ITopDownRuleInductionConfig& setMinCoverage(uint32 minCoverage) except +

        uint32 getMinCoverage() const

        ITopDownRuleInductionConfig& setMaxConditions(uint32 maxConditions) except +

        uint32 getMaxConditions() const;

        ITopDownRuleInductionConfig& setMaxHeadRefinements(uint32 maxHeadRefinements) except +

        uint32 getMaxHeadRefinements() const

        ITopDownRuleInductionConfig& setRecalculatePredictions(bool recalculatePredictions) except +

        bool getRecalculatePredictions() const


cdef class TopDownRuleInductionConfig:

    # Attributes:

    cdef ITopDownRuleInductionConfig* config_ptr
