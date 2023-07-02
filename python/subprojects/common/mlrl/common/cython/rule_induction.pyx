"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less


cdef class GreedyTopDownRuleInductionConfig:
    """
    Allows to configure an algorithm for the induction of individual rules that uses a greedy top-down search.
    """

    def get_min_coverage(self) -> int:
        """
        Returns the minimum number of training examples that must be covered by a rule.

        :return: The minimum number of training examples that must be covered by a rule
        """
        return self.config_ptr.getMinCoverage()

    def set_min_coverage(self, min_coverage: int) -> GreedyTopDownRuleInductionConfig:
        """
        Sets the minimum number of training examples that must be covered by a rule.

        :param min_coverage:    The minimum number of training examples that must be covered by a rule. Must be at least
                                1
        :return:                A `GreedyTopDownRuleInductionConfig` that allows further configuration of the algorithm
                                for the induction of individual rules
        """
        assert_greater_or_equal('min_coverage', min_coverage, 1)
        self.config_ptr.setMinCoverage(min_coverage)
        return self

    def get_min_support(self) -> float:
        """
        Returns the minimum support, i.e., the minimum fraction of the training examples that must be covered by a rule.

        :return: The minimum support or 0, if the support of rules is not restricted
        """
        return self.config_ptr.getMinSupport()

    def set_min_support(self, min_support: float) -> GreedyTopDownRuleInductionConfig:
        """
        Sets the minimum support, i.e., the minimum fraction of the training examples that must be covered by a rule.

        :param min_support: The minimum support. Must be in [0, 1] or 0, if the support of rules should not be
                            restricted
        :return:            A `GreedyTopDownRuleInductionConfig` that allows further configuration of the algorithm for
                            the induction of individual rules
        """
        if min_support != 0:
            assert_greater('min_support', min_support, 0)
            assert_less('min_support', min_support, 1)
        self.config_ptr.setMinSupport(min_support)
        return self

    def get_max_conditions(self) -> int:
        """
        Returns the maximum number of conditions to be included in a rule's body.

        :return: The maximum number of conditions to be included in a rule's body or 0, if the number of conditions is
                 not restricted
        """
        return self.config_ptr.getMaxConditions()

    def set_max_conditions(self, max_conditions: int) -> GreedyTopDownRuleInductionConfig:
        """
        Sets the maximum number of conditions to be included in a rule's body.

        :param max_conditions:  The maximum number of conditions to be included in a rule's body. Must be at least 1 or
                                0, if the number of conditions should not be restricted
        :return:                A `GreedyTopDownRuleInductionConfig` that allows further configuration of the algorithm
                                for the induction of individual rules
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

    def set_max_head_refinements(self, max_head_refinements: int) -> GreedyTopDownRuleInductionConfig:
        """
        Sets the maximum number of times, the head of a rule may be refined after a new condition has been added to its
        body.

        :param max_head_refinements:    The maximum number of times, the head of a rule may be refined. Must be at least
                                        1 or 0, if the number of refinements should not be restricted
        :return:                        A `GreedyTopDownRuleInductionConfig` that allows further configuration of the
                                        algorithm for the induction of individual rules
        """
        if max_head_refinements != 0:
            assert_greater_or_equal('max_head_refinements', max_head_refinements, 1)
        self.config_ptr.setMaxHeadRefinements(max_head_refinements)
        return self

    def are_predictions_recalculated(self) -> bool:
        """
        Returns whether the predictions of rules are recalculated on all training examples, if some of the examples have
        zero weights, or not.

        :return: True, if the predictions of rules are recalculated on all training examples, False otherwise
        """
        return self.config_ptr.arePredictionsRecalculated()

    def set_recalculate_predictions(self, recalculate_predictions: bool) -> GreedyTopDownRuleInductionConfig:
        """
        Sets whether the predictions of rules should be recalculated on all training examples, if some of the examples
        have zero weights, or not.

        :param recalculate_predictions: True, if the predictions of rules should be recalculated on all training
                                        examples, False otherwise
        :return:                        A `GreedyTopDownRuleInductionConfig` that allows further configuration of the
                                        algorithm for the induction of individual rules
        """
        self.config_ptr.setRecalculatePredictions(recalculate_predictions)
        return self


cdef class BeamSearchTopDownRuleInductionConfig:
    """
    Allows to configure an algorithm for the induction of individual rules that uses a top-down beam search.
    """

    def get_beam_width(self) -> int:
        """
        Returns the width that is used by the beam search.

        :return: The width that is used by the beam search
        """
        return self.config_ptr.getBeamWidth()

    def set_beam_width(self, beam_width: int) -> BeamSearchTopDownRuleInductionConfig:
        """
        Sets the width that should be used by the beam search.

        :param beam_width:  The width that should be used by the beam search. Must be at least 2
        :return:            A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of the algorithm
                            for the induction of individual rules
        """
        assert_greater_or_equal('beam_width', beam_width, 2)
        self.config_ptr.setBeamWidth(beam_width)
        return self

    def are_features_resampled(self) -> bool:
        """
         Returns whether a new sample of the available features is created for each rule that is refined during the beam
         search or not.

        :return: True, if a new sample is created for each rule, false otherwise
        """
        return self.config_ptr.areFeaturesResampled()

    def set_resample_features(self, resample_features: bool) -> BeamSearchTopDownRuleInductionConfig:
        """
        Sets whether a new sample of the available features should be created for each rule that is refined during the
        beam search or not.

        :param resample_features:   True, if a new sample should be created for each rule, false otherwise
        :return:                    A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of the
                                    algorithm for the induction of individual rules
        """
        self.config_ptr.setResampleFeatures(resample_features)
        return self

    def get_min_coverage(self) -> int:
        """
        Returns the minimum number of training examples that must be covered by a rule.

        :return: The minimum number of training examples that must be covered by a rule
        """
        return self.config_ptr.getMinCoverage()

    def set_min_coverage(self, min_coverage: int) -> BeamSearchTopDownRuleInductionConfig:
        """
        Sets the minimum number of training examples that must be covered by a rule.

        :param min_coverage:    The minimum number of training examples that must be covered by a rule. Must be at least
                                1
        :return:                A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of the
                                algorithm for the induction of individual rules
        """
        assert_greater_or_equal('min_coverage', min_coverage, 1)
        self.config_ptr.setMinCoverage(min_coverage)
        return self

    def get_min_support(self) -> float:
        """
        Returns the minimum support, i.e., the minimum fraction of the training examples that must be covered by a rule.

        :return: The minimum support or 0, if the support of rules is not restricted
        """
        return self.config_ptr.getMinSupport()

    def set_min_support(self, min_support: float) -> BeamSearchTopDownRuleInductionConfig:
        """
        Sets the minimum support, i.e., the minimum fraction of the training examples that must be covered by a rule.

        :param min_support: The minimum support. Must be in [0, 1] or 0, if the support of rules should not be
                            restricted
        :return:            A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of the algorithm
                            for the induction of individual rules
        """
        if min_support != 0:
            assert_greater('min_support', min_support, 0)
            assert_less('min_support', min_support, 1)
        self.config_ptr.setMinSupport(min_support)
        return self

    def get_max_conditions(self) -> int:
        """
        Returns the maximum number of conditions to be included in a rule's body.

        :return: The maximum number of conditions to be included in a rule's body or 0, if the number of conditions is
                 not restricted
        """
        return self.config_ptr.getMaxConditions()

    def set_max_conditions(self, max_conditions: int) -> BeamSearchTopDownRuleInductionConfig:
        """
        Sets the maximum number of conditions to be included in a rule's body.

        :param max_conditions:  The maximum number of conditions to be included in a rule's body. Must be at least 2 or
                                0, if the number of conditions should not be restricted
        :return:                A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of the
                                algorithm for the induction of individual rules
        """
        if max_conditions != 0:
            assert_greater_or_equal('max_conditions', max_conditions, 2)
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

    def set_max_head_refinements(self, max_head_refinements: int) -> BeamSearchTopDownRuleInductionConfig:
        """
        Sets the maximum number of times, the head of a rule may be refined after a new condition has been added to its
        body.

        :param max_head_refinements:    The maximum number of times, the head of a rule may be refined. Must be at least
                                        1 or 0, if the number of refinements should not be restricted
        :return:                        A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of
                                        the algorithm for the induction of individual rules
        """
        if max_head_refinements != 0:
            assert_greater_or_equal('max_head_refinements', max_head_refinements, 1)
        self.config_ptr.setMaxHeadRefinements(max_head_refinements)
        return self

    def are_predictions_recalculated(self) -> bool:
        """
        Returns whether the predictions of rules are recalculated on all training examples, if some of the examples have
        zero weights, or not.

        :return: True, if the predictions of rules are recalculated on all training examples, False otherwise
        """
        return self.config_ptr.arePredictionsRecalculated()

    def set_recalculate_predictions(self, recalculate_predictions: bool) -> BeamSearchTopDownRuleInductionConfig:
        """
        Sets whether the predictions of rules should be recalculated on all training examples, if some of the examples
        have zero weights, or not.

        :param recalculate_predictions: True, if the predictions of rules should be recalculated on all training
                                        examples, False otherwise
        :return:                        A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of
                                        the algorithm for the induction of individual rules
        """
        self.config_ptr.setRecalculatePredictions(recalculate_predictions)
        return self
