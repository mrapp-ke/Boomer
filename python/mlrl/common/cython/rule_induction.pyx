"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython.head_refinement cimport HeadRefinementFactory
from mlrl.common.cython.post_processing cimport PostProcessor
from mlrl.common.cython.pruning cimport Pruning
from mlrl.common.cython.sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling, PartitionSampling
from mlrl.common.cython.statistics cimport StatisticsProviderFactory
from mlrl.common.cython.stopping cimport StoppingCriterion
from mlrl.common.cython.thresholds cimport ThresholdsFactory

from cython.operator cimport dereference

from libcpp.memory cimport make_unique, make_shared
from libcpp.utility cimport move


cdef class RuleInduction:
    """
    A wrapper for the pure virtual C++ class `IRuleInduction`.
    """
    pass


cdef class TopDownRuleInduction(RuleInduction):
    """
    A wrapper for the C++ class `TopDownRuleInduction`.
    """

    def __cinit__(self, uint32 num_threads):
        """
        :param num_threads: The number of CPU threads to be used to search for potential refinements of a rule in
                            parallel. Must be at least 1
        """
        self.rule_induction_ptr = <shared_ptr[IRuleInduction]>make_shared[TopDownRuleInductionImpl](num_threads)


cdef class RuleModelInduction:
    """
    A wrapper for the pure virtual C++ class `IRuleModelInduction`.
    """

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder):
        cdef shared_ptr[IRuleModelInduction] rule_model_induction_ptr = self.rule_model_induction_ptr
        cdef unique_ptr[RNG] rng_ptr = make_unique[RNG](random_state)
        cdef unique_ptr[RuleModelImpl] rule_model_ptr = rule_model_induction_ptr.get().induceRules(
            nominal_feature_mask.nominal_feature_mask_ptr, feature_matrix.feature_matrix_ptr,
            label_matrix.label_matrix_ptr, dereference(rng_ptr.get()),
            dereference(model_builder.model_builder_ptr.get()))
        cdef RuleModel model = RuleModel()
        model.model_ptr = move(rule_model_ptr)
        return model


cdef class SequentialRuleModelInduction(RuleModelInduction):
    """
    A wrapper for the C++ class `SequentialRuleModelInduction`.
    """

    def __cinit__(self, StatisticsProviderFactory statistics_provider_factory, ThresholdsFactory thresholds_factory,
                  RuleInduction rule_induction, HeadRefinementFactory default_rule_head_refinement_factory,
                  HeadRefinementFactory head_refinement_factory, LabelSubSampling label_sub_sampling,
                  InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                  PartitionSampling partition_sampling, Pruning pruning, PostProcessor post_processor,
                  uint32 min_coverage, intp max_conditions, intp max_head_refinements, list stopping_criteria):
        """
        :param statistics_provider_factory:             A factory that allows to create a provider that provides access
                                                        to the statistics which serve as the basis for learning rules
        :param thresholds_factory:                      A factory that allows to create objects that provide access to
                                                        the thresholds that may be used by the conditions of rules
        :param rule_induction:                          The algorithm that should be used to induce rules
        :param default_rule_head_refinement_factory:    The factory that allows to create instances of the class that
                                                        implements the strategy that should be used to find the head of
                                                        the default rule
        :param head_refinement_factory:                 The factory that allows to create instances of the class that
                                                        implements the strategy that should be used to find the heads of
                                                        rules
        :param label_sub_sampling:                      The strategy that should be used for sub-sampling the labels
                                                        each time a new classification rule is learned
        :param instance_sub_sampling:                   The strategy that should be used for sub-sampling the training
                                                        examples each time a new classification rule is learned
        :param feature_sub_sampling:                    The strategy that should be used for sub-sampling the features
                                                        each time a classification rule is refined
        :param partition_sampling:                      The strategy that should be used for partitioning the training
                                                        examples into a training set and a holdout set or
        :param pruning:                                 The strategy that should be used for pruning rules
        :param post_processor:                          The post-processor that should be used to post-process the rule
                                                        once it has been learned
        :param min_coverage:                            The minimum number of training examples that must be covered by
                                                        a rule. Must be at least 1
        :param max_conditions:                          The maximum number of conditions to be included in a rule's
                                                        body. Must be at least 1 or -1, if the number of conditions
                                                        should not be restricted
        :param max_head_refinements:                    The maximum number of times the head of a rule may be refined
                                                        after a new condition has been added to its body. Must be at
                                                        least 1 or -1, if the number of refinements should not be
                                                        restricted
        :param stopping_criteria                        A list that contains the stopping criteria that should be used
                                                        to decide whether additional rules should be induced or not
        """

        cdef unique_ptr[forward_list[shared_ptr[IStoppingCriterion]]] stopping_criteria_ptr = make_unique[forward_list[shared_ptr[IStoppingCriterion]]]()
        cdef uint32 num_stopping_criteria = len(stopping_criteria)
        cdef StoppingCriterion stopping_criterion
        cdef uint32 i

        for i in range(num_stopping_criteria):
            stopping_criterion = stopping_criteria[i]
            stopping_criteria_ptr.get().push_front(stopping_criterion.stopping_criterion_ptr)

        self.rule_model_induction_ptr = <shared_ptr[IRuleModelInduction]>make_shared[SequentialRuleModelInductionImpl](
            statistics_provider_factory.statistics_provider_factory_ptr, thresholds_factory.thresholds_factory_ptr,
            rule_induction.rule_induction_ptr, default_rule_head_refinement_factory.head_refinement_factory_ptr,
            head_refinement_factory.head_refinement_factory_ptr, label_sub_sampling.label_sub_sampling_ptr,
            instance_sub_sampling.instance_sub_sampling_ptr, feature_sub_sampling.feature_sub_sampling_ptr,
            partition_sampling.partition_sampling_ptr, pruning.pruning_ptr, post_processor.post_processor_ptr,
            min_coverage, max_conditions, max_head_refinements, move(stopping_criteria_ptr))
