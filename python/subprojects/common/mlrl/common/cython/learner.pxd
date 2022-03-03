from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython.feature_binning cimport IEqualWidthFeatureBinningConfig, IEqualFrequencyFeatureBinningConfig
from mlrl.common.cython.feature_matrix cimport IColumnWiseFeatureMatrix, IRowWiseFeatureMatrix
from mlrl.common.cython.feature_sampling cimport IFeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling cimport IExampleWiseStratifiedInstanceSamplingConfig, \
    ILabelWiseStratifiedInstanceSamplingConfig, IInstanceSamplingWithReplacementConfig, \
    IInstanceSamplingWithoutReplacementConfig
from mlrl.common.cython.label_matrix cimport IRowWiseLabelMatrix
from mlrl.common.cython.label_sampling cimport ILabelSamplingWithoutReplacementConfig
from mlrl.common.cython.label_space_info cimport LabelSpaceInfo, ILabelSpaceInfo
from mlrl.common.cython.multi_threading cimport IManualMultiThreadingConfig
from mlrl.common.cython.nominal_feature_mask cimport INominalFeatureMask
from mlrl.common.cython.partition_sampling cimport IExampleWiseStratifiedBiPartitionSamplingConfig, \
    ILabelWiseStratifiedBiPartitionSamplingConfig, IRandomBiPartitionSamplingConfig
from mlrl.common.cython.rule_induction cimport ITopDownRuleInductionConfig
from mlrl.common.cython.rule_model cimport RuleModel, IRuleModel
from mlrl.common.cython.rule_model_assemblage cimport ISequentialRuleModelAssemblageConfig
from mlrl.common.cython.stopping_criterion cimport ISizeStoppingCriterionConfig, ITimeStoppingCriterionConfig, \
    IMeasureStoppingCriterionConfig

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/output/prediction_matrix_dense.hpp" nogil:

    cdef cppclass DensePredictionMatrix[T]:

        # Functions:

        T* release()


cdef extern from "common/output/prediction_matrix_sparse_binary.hpp" nogil:

    cdef cppclass BinarySparsePredictionMatrix:

        # Functions:

        uint32 getNumNonZeroElements() const

        uint32* releaseRowIndices()

        uint32* releaseColIndices()


cdef extern from "common/learner.hpp" nogil:

    cdef cppclass ITrainingResult:

        # Functions:

        uint32 getNumLabels() const

        unique_ptr[IRuleModel]& getRuleModel()

        unique_ptr[ILabelSpaceInfo]& getLabelSpaceInfo()


    cdef cppclass IRuleLearnerConfig"IRuleLearner::IConfig":

        # Functions:

        ISequentialRuleModelAssemblageConfig& useSequentialRuleModelAssemblage()

        ITopDownRuleInductionConfig& useTopDownRuleInduction()

        void useNoFeatureBinning()

        IEqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning()

        IEqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning()

        void useNoLabelSampling()

        ILabelSamplingWithoutReplacementConfig& useLabelSamplingWithoutReplacement()

        void useNoInstanceSampling()

        IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling()

        ILabelWiseStratifiedInstanceSamplingConfig& useLabelWiseStratifiedInstanceSampling()

        IInstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement()

        IInstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement()

        void useNoFeatureSampling()

        IFeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement()

        void useNoPartitionSampling()

        IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling()

        ILabelWiseStratifiedBiPartitionSamplingConfig& useLabelWiseStratifiedBiPartitionSampling()

        IRandomBiPartitionSamplingConfig& useRandomBiPartitionSampling()

        void useNoPruning()

        void useIrepPruning()

        void useNoPostProcessor()

        void useNoParallelRuleRefinement()

        IManualMultiThreadingConfig& useParallelRuleRefinement()

        void useNoParallelStatisticUpdate()

        IManualMultiThreadingConfig& useParallelStatisticUpdate()

        void useNoParallelPrediction()

        IManualMultiThreadingConfig& useParallelPrediction()

        void useNoSizeStoppingCriterion()

        ISizeStoppingCriterionConfig& useSizeStoppingCriterion();

        void useNoTimeStoppingCriterion()

        ITimeStoppingCriterionConfig& useTimeStoppingCriterion();

        void useNoMeasureStoppingCriterion()

        IMeasureStoppingCriterionConfig& useMeasureStoppingCriterion();


    cdef cppclass IRuleLearner:

        # Functions:

        unique_ptr[ITrainingResult] fit(const INominalFeatureMask& nominalFeatureMask,
                                        const IColumnWiseFeatureMatrix& featureMatrix,
                                        const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const

        unique_ptr[DensePredictionMatrix[uint8]] predictLabels(const IRowWiseFeatureMatrix& featureMatrix,
                                                               const IRuleModel& ruleModel,
                                                               const ILabelSpaceInfo& labelSpaceInfo,
                                                               uint32 numLabels) const

        unique_ptr[BinarySparsePredictionMatrix] predictSparseLabels(const IRowWiseFeatureMatrix& featureMatrix,
                                                                     const IRuleModel& ruleModel,
                                                                     const ILabelSpaceInfo& labelSpaceInfo,
                                                                     uint32 numLabels) const

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[DensePredictionMatrix[float64]] predictProbabilities(const IRowWiseFeatureMatrix& featureMatrix,
                                                                        const IRuleModel& ruleModel,
                                                                        const ILabelSpaceInfo& labelSpaceInfo,
                                                                        uint32 numLabels) const


cdef class TrainingResult:

    # Attributes:

    cdef readonly uint32 num_labels

    cdef readonly RuleModel rule_model

    cdef readonly LabelSpaceInfo label_space_info


cdef class RuleLearnerConfig:

    # Functions:

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self)


cdef class RuleLearner:

    # Functions:

    cdef IRuleLearner* get_rule_learner_ptr(self)
