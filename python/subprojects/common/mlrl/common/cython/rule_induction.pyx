"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal


cdef class TopDownRuleInductionConfig:
    """
    Allows to configure an algorithm for the induction of individual rules that uses a top-down greedy search.
    """

    def get_min_coverage(self) -> int:
        """
        Returns the minimum number of training examples that must be covered by a rule.

        :return: The minimum number of training examples that must be covered by a rule.
        """
        return self.config_ptr.getMinCoverage()

    def set_min_coverage(self, min_coverage: int) -> TopDownRuleInductionConfig:
        """
        Sets the minimum number of training examples that must be covered by a rule.

        :param min_coverage:    The minimum number of training examples that must be covered by a rule. Must be at least
                                1
        :return:                A `TopDownRuleInductionConfig` that allows further configuration of the algorithm for
                                the induction of individual rules
        """
        assert_greater_or_equal('min_coverage', min_coverage, 1)
        self.config_ptr.setMinCoverage(min_coverage)
        return self

    def get_max_conditions(self) -> int:
        """
        Returns the maximum number of conditions to be included in a rule's body.

        :return: The maximum number of conditions to be included in a rule's body or 0, if the number of conditions is
                 not restricted
        """
        return self.config_ptr.getMaxConditions()

    def set_max_conditions(self, max_conditions: int) -> TopDownRuleInductionConfig:
        """
        Sets the maximum number of conditions to be included in a rule's body.

        :param max_conditions:  The maximum number of conditions to be included in a rule's body. Must be at least 1 or
                                0, if the number of conditions should not be restricted
        :return:                A `TopDownRuleInductionConfig` that allows further configuration of the algorithm for
                                the induction of individual rules
        """
        if max_conditions != 0:
            assert_greater_or_equal('max_conditions', max_conditions, 1)
        self.config_ptr.setMaxConditions(max_conditions)
        return self

    def get_max_head_refinements(self) -> int:
        """
        Returns the maximum number of times, the head of a rule may be refinement after a new condition has been added
        to its body.

        :return: The maximum number of times, the head of a rule may be refined or 0, if the number of refinements is
                 not restricted
        """
        return self.config_ptr.getMaxHeadRefinements()

    def set_max_head_refinements(self, max_head_refinements: int) -> TopDownRuleInductionConfig:
        """
        Sets the maximum number of times, the head of a rule may be refined after a new condition has been added to its
        body.

        :param max_head_refinements:    The maximum number of times, the head of a rule may be refined. Must be at least
                                        1 or 0, if the number of refinements should not be restricted
        :return:                        A `TopDownRuleInductionConfig` that allows further configuration of the
                                        algorithm for the induction of individual rules
        """
        if max_head_refinements != 0:
            assert_greater_or_equal('max_head_refinements', max_head_refinements, 1)
        self.config_ptr.setMaxHeadRefinements(max_head_refinements)
        return self

    def get_recalculate_predictions(self) -> bool:
        """
        Returns whether the predictions of rules are recalculated on all training examples, if some of the examples have
        zero weights, or not.

        :return: True, if the predictions of rules are recalculated on all training examples, False otherwise
        """
        return self.config_ptr.getRecalculatePredictions()

    def set_recalculate_predictions(self, recalculate_predictions: bool) -> TopDownRuleInductionConfig:
        """
        Sets whether the predictions of rules should be recalculated on all training examples, if some of the examples
        have zero weights, or not.

        :param recalculate_predictions: True, if the predictions of rules should be recalculated on all training
                                        examples, False otherwise
        :return:                        A `TopDownRuleInductionConfig` that allows further configuration of the
                                        algorithm for the induction of individual rules
        """
        self.config_ptr.setRecalculatePredictions(recalculate_predictions)
        return self
