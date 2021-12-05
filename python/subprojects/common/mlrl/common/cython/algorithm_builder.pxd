from mlrl.common.cython.statistics cimport IStatisticsProviderFactory
from mlrl.common.cython.thresholds cimport IThresholdsFactory
from mlrl.common.cython.rule_induction cimport IRuleInduction
from mlrl.common.cython.feature_sampling cimport IFeatureSamplingFactory
from mlrl.common.cython.instance_sampling cimport IInstanceSamplingFactory
from mlrl.common.cython.label_sampling cimport ILabelSamplingFactory
from mlrl.common.cython.partition_sampling cimport IPartitionSamplingFactory
from mlrl.common.cython.pruning cimport IPruning
from mlrl.common.cython.post_processing cimport IPostProcessor
from mlrl.common.cython.stopping cimport IStoppingCriterion
from mlrl.common.cython.rule_model_assemblage cimport IRuleModelAssemblage, IRuleModelAssemblageFactory

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/algorithm_builder.hpp" nogil:

    cdef cppclass AlgorithmBuilderImpl"AlgorithmBuilder":

        # Constructor:

        AlgorithmBuilderImpl(unique_ptr[IStatisticsProviderFactory] statisticsProviderFactoryPtr,
                             unique_ptr[IThresholdsFactory] thresholdsFactoryPtr,
                             unique_ptr[IRuleInduction] ruleInductionPtr,
                             unique_ptr[IRuleModelAssemblageFactory] ruleModelAssemblageFactoryPtr) except +

        # Functions:

        AlgorithmBuilderImpl& setUseDefaultRule(bool useDefaultRule) except +

        AlgorithmBuilderImpl& setLabelSamplingFactory(
            unique_ptr[ILabelSamplingFactory] labelSamplingFactoryPtr) except +

        AlgorithmBuilderImpl& setInstanceSamplingFactory(
            unique_ptr[IInstanceSamplingFactory] instanceSamplingFactoryPtr) except +

        AlgorithmBuilderImpl& setFeatureSamplingFactory(
            unique_ptr[IFeatureSamplingFactory] featureSamplingFactoryPtr) except +

        AlgorithmBuilderImpl& setPartitionSamplingFactory(
            unique_ptr[IPartitionSamplingFactory] partitionSamplingFactoryPtr) except +

        AlgorithmBuilderImpl& setPruning(unique_ptr[IPruning] pruningPtr) except +

        AlgorithmBuilderImpl& setPostProcessor(unique_ptr[IPostProcessor] postProcessorPtr) except +

        AlgorithmBuilderImpl& addStoppingCriterion(unique_ptr[IStoppingCriterion] stoppingCriterionPtr) except +

        unique_ptr[IRuleModelAssemblage] build() const


cdef class AlgorithmBuilder:

    # Attributes:

    cdef unique_ptr[AlgorithmBuilderImpl] builder_ptr
