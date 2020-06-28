"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to sequentially induce models that consist of several classification rules.
"""
from boomer.common._random cimport RNG
from boomer.common.rules cimport Rule, RuleList
from boomer.common.stopping_criteria cimport StoppingCriterion


cdef class SequentialRuleInduction:
    """
    Allows to sequentially induce classification rules, including a default rule, that are added to a model using a
    `ModelBuilder`.
    """

    def __cinit__(self, RuleInduction rule_induction, HeadRefinement head_refinement, Loss loss, list stopping_criteria,
                  LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                  FeatureSubSampling feature_sub_sampling, Pruning pruning, Shrinkage shrinkage, intp min_coverage,
                  intp max_conditions, intp max_head_refinements):
        """
        :param rule_induction:          The algorithm that should be used to induce rules
        :param head_refinement:         The strategy that should be used to find the heads of rules
        :param loss:                    The loss function to be minimized
        :param stopping_criteria        A list that contains the stopping criteria that should be used to decide whether
                                        additional rules should be induced or not
        :param label_sub_sampling:      The strategy that should be used for sub-sampling the labels each time a new
                                        classification rule is learned or None, if no sub-sampling should be used
        :param instance_sub_sampling:   The strategy that should be used for sub-sampling the training examples each
                                        time a new classification rule is learned or None, if no sub-sampling should be
                                        used
        :param feature_sub_sampling:    The strategy that should be used for sub-sampling the features each time a
                                        classification rule is refined or None, if no sub-sampling should be used
        :param pruning:                 The strategy that should be used for pruning rules or None, if no pruning should
                                        be used
        :param shrinkage:               The strategy that should be used for shrinking the weights of rule, or None if
                                        no shrinkage should be used
        :param min_coverage:            The minimum number of training examples that must be covered by a rule. Must be
                                        at least 1
        :param max_conditions:          The maximum number of conditions to be included in a rule's body. Must be at
                                        least 1 or -1, if the number of conditions should not be restricted
        :param max_head_refinements:    The maximum number of times the head of a rule may be refined after a new
                                        condition has been added to its body. Must be at least 1 or -1, if the number of
                                        refinements should not be restricted
        """
        self.rule_induction = rule_induction
        self.head_refinement = head_refinement
        self.loss = loss
        self.stopping_criteria = stopping_criteria
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements

    cpdef RuleModel induce_rules(self, intp[::1] nominal_attribute_indices, FeatureMatrix feature_matrix,
                                 uint8[::1, :] y, uint32 random_state, ModelBuilder model_builder):
        """
        Creates and returns a model that consists of several classification rules.

        :param nominal_attribute_indices:   An array of dtype int, shape `(num_nominal_features)`, representing the
                                            indices of all nominal attributes (in ascending order)
        :param feature_matrix:              The `FeatureMatrix` that provides column-wise access to the feature values
                                            of the training examples
        :param y:                           An array of dtype int, shape `(num_examples, num_labels)`, representing
                                            the labels of the training examples
        :param random_state:                The seed to be used by RNGs
        :param model_builder:               The builder that should be used to build the model
        :return:                            A model that contains the induced classification rules
        """
        cdef RuleInduction rule_induction = self.rule_induction
        cdef HeadRefinement head_refinement = self.head_refinement
        cdef Loss loss = self.loss
        cdef list stopping_criteria = self.stopping_criteria
        cdef LabelSubSampling label_sub_sampling = self.label_sub_sampling
        cdef InstanceSubSampling instance_sub_sampling = self.instance_sub_sampling
        cdef FeatureSubSampling feature_sub_sampling = self.feature_sub_sampling
        cdef Pruning pruning = self.pruning
        cdef Shrinkage shrinkage = self.shrinkage
        cdef intp min_coverage = self.min_coverage
        cdef intp max_conditions = self.max_conditions
        cdef intp max_head_refinements = self.max_head_refinements
        cdef RNG rng = RNG.__new__(RNG, random_state)
        # The total number of labels
        cdef intp num_labels = y.shape[1]
        # The number of rules induced so far (starts at 1 to account for the default rule)
        cdef intp num_rules = 1
        # Temporary variables
        cdef bint success

        # Induce default rule...
        rule_induction.induce_default_rule(y, loss, model_builder)

        while __should_continue(stopping_criteria, num_rules):
            # Induce a new rule...
            success = rule_induction.induce_rule(nominal_attribute_indices, feature_matrix, num_labels, head_refinement,
                                                 loss, label_sub_sampling, instance_sub_sampling, feature_sub_sampling,
                                                 pruning, shrinkage, min_coverage, max_conditions, max_head_refinements,
                                                 rng, model_builder)

            if not success:
                break

            num_rules += 1

        return model_builder.build_model()


cdef inline bint __should_continue(list stopping_criteria, intp num_rules):
    """
    Returns whether additional rules should be induced, according to some stopping criteria, or not.

    :param stopping_criteria:   A list that contains the stopping criteria that should be used to decide whether
                                additional rules should be induced or not
    :param num_rules:           The number of rules induced so far (including the default rule)
    :return:                    True, if additional rules should be induced, False otherwise
    """
    cdef StoppingCriterion stopping_criterion

    for stopping_criterion in stopping_criteria:
        if not stopping_criterion.should_continue(num_rules):
            return False

    return True
