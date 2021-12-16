"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class RuleInduction:
    """
    A wrapper for the pure virtual C++ class `IRuleInduction`.
    """
    pass


cdef class TopDownRuleInduction(RuleInduction):
    """
    A wrapper for the C++ class `TopDownRuleInduction`.
    """

    def __cinit__(self, uint32 min_coverage, uint32 max_conditions, uint32 max_head_refinements,
                  bint recalculate_predictions, uint32 num_threads):
        """
        :param min_coverage:            The minimum number of training examples that must be covered by a rule. Must be
                                        at least 1
        :param max_conditions:          The maximum number of conditions to be included in a rule's body. Must be at
                                        least 1 or -1, if the number of conditions should not be restricted
        :param max_head_refinements:    The maximum number of times the head of a rule may be refined after a new
                                        condition has been added to its body. Must be at least 1 or -1, if the number of
                                        refinements should not be restricted
        :param recalculate_predictions: True, if the predictions of rules should be recalculated on the entire training
                                        data, if instance sampling is used, False otherwise
        :param num_threads:             The number of CPU threads to be used to search for potential refinements of a
                                        rule in parallel. Must be at least 1
        """
        self.rule_induction_ptr = <unique_ptr[IRuleInduction]>make_unique[TopDownRuleInductionImpl](
            min_coverage, max_conditions, max_head_refinements, recalculate_predictions, num_threads)
