from mlrl.common.cython._types cimport uint32
from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/rule_induction/rule_induction.hpp" nogil:

    cdef cppclass IRuleInduction:
        pass


cdef extern from "common/rule_induction/rule_induction_top_down.hpp" nogil:

    cdef cppclass TopDownRuleInductionImpl"TopDownRuleInduction"(IRuleInduction):

        # Constructors:

        TopDownRuleInductionImpl(uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                                 bool recalculatePredictions, uint32 numThreads) except +


cdef class RuleInduction:

    # Attributes:

    cdef unique_ptr[IRuleInduction] rule_induction_ptr


cdef class TopDownRuleInduction(RuleInduction):
    pass
