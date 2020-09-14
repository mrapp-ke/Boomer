from boomer.common._arrays cimport uint8, uint32, intp
from boomer.common.rules cimport RuleModel, ModelBuilder
from boomer.common.rule_induction cimport RuleInduction
from boomer.common.statistics cimport StatisticsProviderFactory
from boomer.common.head_refinement cimport HeadRefinement
from boomer.common.input_data cimport LabelMatrix, FeatureMatrix
from boomer.common.pruning cimport Pruning
from boomer.common.post_processing cimport PostProcessor
from boomer.common.sub_sampling cimport InstanceSubSampling, FeatureSubSampling, LabelSubSampling


cdef class SequentialRuleInduction:

    # Attributes:

    cdef StatisticsProviderFactory statistics_provider_factory

    cdef RuleInduction rule_induction

    cdef HeadRefinement default_rule_head_refinement

    cdef HeadRefinement head_refinement

    cdef list stopping_criteria

    cdef LabelSubSampling label_sub_sampling

    cdef InstanceSubSampling instance_sub_sampling

    cdef FeatureSubSampling feature_sub_sampling

    cdef Pruning pruning

    cdef PostProcessor post_processor

    cdef uint32 min_coverage

    cdef intp max_conditions

    cdef intp max_head_refinements

    cdef int num_threads

    # Functions:

    cpdef RuleModel induce_rules(self, uint8[::1] nominal_attribute_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder)
