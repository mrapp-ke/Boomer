from mlrl.common.cython._types cimport uint32, intp
from mlrl.common.cython.input cimport NominalFeatureMask, INominalFeatureMask
from mlrl.common.cython.input cimport FeatureMatrix, IFeatureMatrix
from mlrl.common.cython.input cimport LabelMatrix, ILabelMatrix
from mlrl.common.cython.model cimport ModelBuilder, RuleModel, IModelBuilder, RuleModelImpl
from mlrl.common.cython.sampling cimport ILabelSubSampling, IInstanceSubSampling, IFeatureSubSampling, \
    IPartitionSampling, RNG
from mlrl.common.cython.statistics cimport IStatisticsProviderFactory
from mlrl.common.cython.stopping cimport IStoppingCriterion
from mlrl.common.cython.thresholds cimport IThresholdsFactory
from mlrl.common.cython.pruning cimport IPruning
from mlrl.common.cython.post_processing cimport IPostProcessor
from mlrl.common.cython.head_refinement cimport IHeadRefinementFactory

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.forward_list cimport forward_list


cdef extern from "common/rule_induction/rule_induction.hpp" nogil:

    cdef cppclass IRuleInduction:
        pass


cdef extern from "common/rule_induction/rule_model_induction.hpp" nogil:

    cdef cppclass IRuleModelInduction:

        # Functions:

        unique_ptr[RuleModelImpl] induceRules(shared_ptr[INominalFeatureMask] nominalFeatureMaskPtr,
                                              shared_ptr[IFeatureMatrix] featureMatrixPtr,
                                              shared_ptr[ILabelMatrix] labelMatrixPtr, RNG& rng,
                                              IModelBuilder& modelBuilder)


cdef extern from "common/rule_induction/rule_induction_top_down.hpp" nogil:

    cdef cppclass TopDownRuleInductionImpl"TopDownRuleInduction"(IRuleInduction):

        # Constructors:

        TopDownRuleInductionImpl(uint32 numThreads) except +


cdef extern from "common/rule_induction/rule_model_induction_sequential.hpp" nogil:

    cdef cppclass SequentialRuleModelInductionImpl"SequentialRuleModelInduction"(IRuleModelInduction):

        # Constructors:

        SequentialRuleModelInductionImpl(shared_ptr[IStatisticsProviderFactory] statisticsProviderFactoryPtr,
                                         shared_ptr[IThresholdsFactory] thresholdsFactoryPtr,
                                         shared_ptr[IRuleInduction] ruleInductionPtr,
                                         shared_ptr[IHeadRefinementFactory] defaultRuleHeadRefinementFactoryPtr,
                                         shared_ptr[IHeadRefinementFactory] headRefinementFactoryPtr,
                                         shared_ptr[ILabelSubSampling] labelSubSamplingPtr,
                                         shared_ptr[IInstanceSubSampling] instanceSubSamplingPtr,
                                         shared_ptr[IFeatureSubSampling] featureSubSamplingPtr,
                                         shared_ptr[IPartitionSampling] partitionSamplingPtr,
                                         shared_ptr[IPruning] pruningPtr, shared_ptr[IPostProcessor] postProcessorPtr,
                                         uint32 minCoverage, intp maxConditions, intp maxHeadRefinements,
                                         unique_ptr[forward_list[shared_ptr[IStoppingCriterion]]] stoppingCriteriaPtr) except +


cdef class RuleInduction:

    # Attributes:

    cdef shared_ptr[IRuleInduction] rule_induction_ptr


cdef class TopDownRuleInduction(RuleInduction):
    pass


cdef class RuleModelInduction:

    # Attributes:

    cdef shared_ptr[IRuleModelInduction] rule_model_induction_ptr

    # Functions:

    cpdef RuleModel induce_rules(self, NominalFeatureMask nominal_feature_mask, FeatureMatrix feature_matrix,
                                 LabelMatrix label_matrix, uint32 random_state, ModelBuilder model_builder)



cdef class SequentialRuleModelInduction(RuleModelInduction):
    pass
