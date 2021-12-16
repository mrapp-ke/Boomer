"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.feature_sampling cimport FeatureSamplingFactory
from mlrl.common.cython.instance_sampling cimport InstanceSamplingFactory
from mlrl.common.cython.label_sampling cimport LabelSamplingFactory
from mlrl.common.cython.partition_sampling cimport PartitionSamplingFactory
from mlrl.common.cython.pruning cimport Pruning
from mlrl.common.cython.post_processing cimport PostProcessor
from mlrl.common.cython.rule_induction cimport RuleInduction
from mlrl.common.cython.rule_model_assemblage cimport RuleModelAssemblage, RuleModelAssemblageFactory
from mlrl.common.cython.statistics cimport StatisticsProviderFactory
from mlrl.common.cython.stopping cimport StoppingCriterion
from mlrl.common.cython.thresholds cimport ThresholdsFactory

from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class AlgorithmBuilder:
    """
    A wrapper for the C++ class `AlgorithmBuilder`.
    """

    def __cinit__(self, StatisticsProviderFactory statistics_provider_factory not None,
                  ThresholdsFactory thresholds_factory not None, RuleInduction rule_induction not None,
                  RuleModelAssemblageFactory rule_model_assemblage_factory not None):
        """
        :param statistics_provider_factory:     The `StatisticsProviderFactory` to be used by the rule learner to access
                                                the statistics that serve as the basis for learning rules
        :param thresholds_factory:              The `IThresholdsFactory` to be used by the rule learner to access the
                                                thresholds that may be used by the conditions of rules
        :param rule_induction:                  The `IRuleInduction` to be used by the rule learner to induce individual
                                                rules
        :param rule_model_assemblage_factory:   The `IRuleModelAssemblageFactory` to be used by the rule learner for the
                                                assemblage of a rule model
        """
        self.builder_ptr = make_unique[AlgorithmBuilderImpl](
            move(statistics_provider_factory.statistics_provider_factory_ptr),
            move(thresholds_factory.thresholds_factory_ptr), move(rule_induction.rule_induction_ptr),
            move(rule_model_assemblage_factory.rule_model_assemblage_factory_ptr))

    def set_use_default_rule(self, bint use_default_rule) -> AlgorithmBuilder:
        """
        Sets whether a default rule be used by the rule learner or not.

        :param default_rule:    True, if a default rule should be used, False otherwise
        :return                 The builder itself
        """
        self.builder_ptr.get().setUseDefaultRule(use_default_rule);
        return self

    def set_label_sampling_factory(self, LabelSamplingFactory label_sampling_factory not None) -> AlgorithmBuilder:
        """
        Sets the `LabelSamplingFactory` to be used by the rule learner to sample the labels individual rules may predict
        for.

        :param label_sampling_factory:  The `LabelSamplingFactory` to be set
        :return:                        The builder itself
        """
        self.builder_ptr.get().setLabelSamplingFactory(move(label_sampling_factory.label_sampling_factory_ptr))
        return self

    def set_instance_sampling_factory(self,
                                      InstanceSamplingFactory instance_sampling_factory not None) -> AlgorithmBuilder:
        """
        Sets the `InstanceSamplingFactory` to be used by the rule learner to sample the instances whenever a new rule is
        induced.

        :param instance_sampling_factory:   The `InstanceSamplingFactory` to be set
        :return:                            The builder itself
        """
        self.builder_ptr.get().setInstanceSamplingFactory(move(instance_sampling_factory.instance_sampling_factory_ptr))
        return self

    def set_feature_sampling_factory(self,
                                     FeatureSamplingFactory feature_sampling_factory not None) -> AlgorithmBuilder:
        """
        Sets the `FeatureSamplingFactory` to be used by the rule learner to sample the features whenever a rule should
        be refined.

        :param feature_sampling_factory:    The `FeatureSamplingFactory` to be set
        :return:                            The builder itself
        """
        self.builder_ptr.get().setFeatureSamplingFactory(move(feature_sampling_factory.feature_sampling_factory_ptr))
        return self

    def set_partition_sampling_factory(
            self, PartitionSamplingFactory partition_sampling_factory not None) -> AlgorithmBuilder:
        """
        Sets the `PartitionSamplingFactory` to be used by the rule learner to create a holdout set.

        :param partition_sampling_factory:  The `PartitionSamplingFactory` to be set
        :return:                            The builder itself
        """
        self.builder_ptr.get().setPartitionSamplingFactory(move(partition_sampling_factory.partition_sampling_factory_ptr))
        return self

    def set_pruning(self, Pruning pruning not None ) -> AlgorithmBuilder:
        """
        Sets the `Pruning` to be used by the rule learner to prune individual rules.

        :param pruning: The `Pruning` to be set
        :return:        The builder itself
        """
        self.builder_ptr.get().setPruning(move(pruning.pruning_ptr))
        return self

    def set_post_processor(self, PostProcessor post_processor not None) -> AlgorithmBuilder:
        """
        Sets the `PostProcessor` to be used by the rule learner to post-process the predictions of individual rules.

        :param post_processor:  The `PostProcessor` to be set
        :return:                The builder itself
        """
        self.builder_ptr.get().setPostProcessor(move(post_processor.post_processor_ptr))
        return self

    def add_stopping_criterion(self, StoppingCriterion stopping_criterion not None) -> AlgorithmBuilder:
        """
        Adds a `StoppingCriterion` that should be used by the rule learner to decide when the induction of additional
        rules should be stopped.

        :param stopping_criterion:  The `StoppingCriterion` to be added
        :return:                    The builder itself
        """
        self.builder_ptr.get().addStoppingCriterion(move(stopping_criterion.stopping_criterion_ptr))
        return self

    def build(self) -> RuleModelAssemblage:
        """
        Creates and returns a new object of type `RuleModelAssemblage`.

        :return: The object of type `IRuleModelAssemblage` that has been created
        """
        cdef RuleModelAssemblage rule_model_assemblage = RuleModelAssemblage.__new__(RuleModelAssemblage)
        rule_model_assemblage.rule_model_assemblage_ptr = move(self.builder_ptr.get().build())
        return rule_model_assemblage
