from libcpp cimport bool

from mlrl.common.cython._types cimport float32, uint32


cdef extern from "common/rule_induction/rule_induction_top_down_greedy.hpp" nogil:

    cdef cppclass IGreedyTopDownRuleInductionConfig:

        # Functions:

        IGreedyTopDownRuleInductionConfig& setMinCoverage(uint32 minCoverage) except +

        uint32 getMinCoverage() const

        IGreedyTopDownRuleInductionConfig& setMinSupport(float32 minSupport) except +

        float32 getMinSupport() const

        IGreedyTopDownRuleInductionConfig& setMaxConditions(uint32 maxConditions) except +

        uint32 getMaxConditions() const;

        IGreedyTopDownRuleInductionConfig& setMaxHeadRefinements(uint32 maxHeadRefinements) except +

        uint32 getMaxHeadRefinements() const

        IGreedyTopDownRuleInductionConfig& setRecalculatePredictions(bool recalculatePredictions) except +

        bool arePredictionsRecalculated() const


cdef extern from "common/rule_induction/rule_induction_top_down_beam_search.hpp" nogil:

    cdef cppclass IBeamSearchTopDownRuleInductionConfig:

        # Functions:

        IBeamSearchTopDownRuleInductionConfig& setBeamWidth(uint32 beamWidth) except +

        uint32 getBeamWidth() const

        IBeamSearchTopDownRuleInductionConfig& setResampleFeatures(bool resampleFeatures) except +

        bool areFeaturesResampled() const

        IBeamSearchTopDownRuleInductionConfig& setMinCoverage(uint32 minCoverage) except +

        uint32 getMinCoverage() const

        IBeamSearchTopDownRuleInductionConfig& setMinSupport(float32 minSupport) except +

        float32 getMinSupport() const

        IBeamSearchTopDownRuleInductionConfig& setMaxConditions(uint32 maxConditions) except +

        uint32 getMaxConditions() const;

        IBeamSearchTopDownRuleInductionConfig& setMaxHeadRefinements(uint32 maxHeadRefinements) except +

        uint32 getMaxHeadRefinements() const

        IBeamSearchTopDownRuleInductionConfig& setRecalculatePredictions(bool recalculatePredictions) except +

        bool arePredictionsRecalculated() const


cdef class GreedyTopDownRuleInductionConfig:

    # Attributes:

    cdef IGreedyTopDownRuleInductionConfig* config_ptr


cdef class BeamSearchTopDownRuleInductionConfig:

    # Attributes:

    cdef IBeamSearchTopDownRuleInductionConfig* config_ptr
