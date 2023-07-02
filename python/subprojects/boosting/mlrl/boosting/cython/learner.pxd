from mlrl.boosting.cython.head_type cimport IDynamicPartialHeadConfig, IFixedPartialHeadConfig
from mlrl.boosting.cython.label_binning cimport IEqualWidthLabelBinningConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig
from mlrl.boosting.cython.prediction cimport IExampleWiseBinaryPredictorConfig, IGfmBinaryPredictorConfig, \
    ILabelWiseBinaryPredictorConfig, ILabelWiseProbabilityPredictorConfig, IMarginalizedProbabilityPredictorConfig
from mlrl.boosting.cython.probability_calibration cimport IIsotonicJointProbabilityCalibratorConfig, \
    IIsotonicMarginalProbabilityCalibratorConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig

ctypedef double (*DdotFunction)(int* n, double* dx, int* incx, double* dy, int* incy)

ctypedef void (*DspmvFunction)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta,
                               double* y, int* incy)

ctypedef void (*DsysvFunction)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb,
                               double* work, int* lwork, int* info)


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IAutomaticPartitionSamplingMixin"boosting::IBoostingRuleLearner::IAutomaticPartitionSamplingMixin":

        # Functions:

        void useAutomaticPartitionSampling()


    cdef cppclass IAutomaticFeatureBinningMixin"boosting::IBoostingRuleLearner::IAutomaticFeatureBinningMixin":

        # Functions

        void useAutomaticFeatureBinning()


    cdef cppclass IAutomaticParallelRuleRefinementMixin\
        "boosting::IBoostingRuleLearner::IAutomaticParallelRuleRefinementMixin":

        # Functions:

        void useAutomaticParallelRuleRefinement()


    cdef cppclass IAutomaticParallelStatisticUpdateMixin\
        "boosting::IBoostingRuleLearner::IAutomaticParallelStatisticUpdateMixin":

        # Functions:

        void useAutomaticParallelStatisticUpdate()


    cdef cppclass IConstantShrinkageMixin"boosting::IBoostingRuleLearner::IConstantShrinkageMixin":

        # Functions:

        IConstantShrinkageConfig& useConstantShrinkagePostProcessor()


    cdef cppclass INoL1RegularizationMixin"boosting::IBoostingRuleLearner::INoL1RegularizationMixin":

        # Functions:

        void useNoL1Regularization()


    cdef cppclass IL1RegularizationMixin"boosting::IBoostingRuleLearner::IL1RegularizationMixin":

        # Functions:

        IManualRegularizationConfig& useL1Regularization()


    cdef cppclass INoL2RegularizationMixin"boosting::IBoostingRuleLearner::INoL2RegularizationMixin":

        # Functions:

        void useNoL2Regularization()


    cdef cppclass IL2RegularizationMixin"boosting::IBoostingRuleLearner::IL2RegularizationMixin":

        # Functions:

        IManualRegularizationConfig& useL2Regularization()


    cdef cppclass INoDefaultRuleMixin"boosting::IBoostingRuleLearner::INoDefaultRuleMixin":

        # Functions:

        void useNoDefaultRule()


    cdef cppclass IAutomaticDefaultRuleMixin"boosting::IBoostingRuleLearner::IAutomaticDefaultRuleMixin":

        # Functions:

        void useAutomaticDefaultRule()


    cdef cppclass ICompleteHeadMixin"boosting::IBoostingRuleLearner::ICompleteHeadMixin":

        # Functions:

        void useCompleteHeads()


    cdef cppclass IFixedPartialHeadMixin"boosting::IBoostingRuleLearner::IFixedPartialHeadMixin":

        # Functions:

        IFixedPartialHeadConfig& useFixedPartialHeads()


    cdef cppclass IDynamicPartialHeadMixin"boosting::IBoostingRuleLearner::IDynamicPartialHeadMixin":

        # Functions:

        IDynamicPartialHeadConfig& useDynamicPartialHeads()


    cdef cppclass ISingleLabelHeadMixin"boosting::IBoostingRuleLearner::ISingleLabelHeadMixin":

        # Functions:
        
        void useSingleLabelHeads()


    cdef cppclass IAutomaticHeadMixin"boosting::IBoostingRuleLearner::IAutomaticHeadMixin":

        # Functions:

        void useAutomaticHeads()


    cdef cppclass IDenseStatisticsMixin"boosting::IBoostingRuleLearner::IDenseStatisticsMixin":

        # Functions:

        void useDenseStatistics()


    cdef cppclass ISparseStatisticsMixin"boosting::IBoostingRuleLearner::ISparseStatisticsMixin":

        # Functions:

        void useSparseStatistics()


    cdef cppclass IAutomaticStatisticsMixin"boosting::IBoostingRuleLearner::IAutomaticStatisticsMixin":

        # Functions:

        void useAutomaticStatistics()


    cdef cppclass IExampleWiseLogisticLossMixin"boosting::IBoostingRuleLearner::IExampleWiseLogisticLossMixin":

        # Functions:

        void useExampleWiseLogisticLoss()


    cdef cppclass IExampleWiseSquaredErrorLossMixin"boosting::IBoostingRuleLearner::IExampleWiseSquaredErrorLossMixin":

        # Functions:

        void useExampleWiseSquaredErrorLoss()


    cdef cppclass IExampleWiseSquaredHingeLossMixin"boosting::IBoostingRuleLearner::IExampleWiseSquaredHingeLossMixin":

        # Functions:

        void useExampleWiseSquaredHingeLoss()


    cdef cppclass ILabelWiseLogisticLossMixin"boosting::IBoostingRuleLearner::ILabelWiseLogisticLossMixin":

        # Functions:

        void useLabelWiseLogisticLoss()

        
    cdef cppclass ILabelWiseSquaredErrorLossMixin"boosting::IBoostingRuleLearner::ILabelWiseSquaredErrorLossMixin":

        # Functions:

        void useLabelWiseSquaredErrorLoss()


    cdef cppclass ILabelWiseSquaredHingeLossMixin"boosting::IBoostingRuleLearner::ILabelWiseSquaredHingeLossMixin":

        # Functions:

        void useLabelWiseSquaredHingeLoss()


    cdef cppclass INoLabelBinningMixin"boosting::IBoostingRuleLearner::INoLabelBinningMixin":

        # Functions:

        void useNoLabelBinning()


    cdef cppclass IEqualWidthLabelBinningMixin"boosting::IBoostingRuleLearner::IEqualWidthLabelBinningMixin":

        # Functions:

        IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning()


    cdef cppclass IIsotonicMarginalProbabilityCalibrationMixin"boosting::IBoostingRuleLearner::IIsotonicMarginalProbabilityCalibrationMixin":

        # Functions:

        IIsotonicMarginalProbabilityCalibratorConfig& useIsotonicMarginalProbabilityCalibration()


    cdef cppclass IIsotonicJointProbabilityCalibrationMixin"boosting::IBoostingRuleLearner::IIsotonicJointProbabilityCalibrationMixin":

        # Functions:

        IIsotonicJointProbabilityCalibratorConfig& useIsotonicJointProbabilityCalibration()
        

    cdef cppclass IAutomaticLabelBinningMixin"boosting::IBoostingRuleLearner::IAutomaticLabelBinningMixin":

        # Functions:

        void useAutomaticLabelBinning()


    cdef cppclass ILabelWiseBinaryPredictorMixin"boosting::IBoostingRuleLearner::ILabelWiseBinaryPredictorMixin":

        # Functions:

        ILabelWiseBinaryPredictorConfig& useLabelWiseBinaryPredictor()


    cdef cppclass IExampleWiseBinaryPredictorMixin"boosting::IBoostingRuleLearner::IExampleWiseBinaryPredictorMixin":

        # Functions:

        IExampleWiseBinaryPredictorConfig& useExampleWiseBinaryPredictor()


    cdef cppclass IGfmBinaryPredictorMixin"boosting::IBoostingRuleLearner::IGfmBinaryPredictorMixin":

        # Functions:

        IGfmBinaryPredictorConfig& useGfmBinaryPredictor()


    cdef cppclass IAutomaticBinaryPredictorMixin"boosting::IBoostingRuleLearner::IAutomaticBinaryPredictorMixin":

        # Functions:

        void useAutomaticBinaryPredictor()


    cdef cppclass ILabelWiseScorePredictorMixin"boosting::IBoostingRuleLearner::ILabelWiseScorePredictorMixin":

        # Functions:

        void useLabelWiseScorePredictor()


    cdef cppclass ILabelWiseProbabilityPredictorMixin \
        "boosting::IBoostingRuleLearner::ILabelWiseProbabilityPredictorMixin":

        # Functions:

        ILabelWiseProbabilityPredictorConfig& useLabelWiseProbabilityPredictor()


    cdef cppclass IMarginalizedProbabilityPredictorMixin\
        "boosting::IBoostingRuleLearner::IMarginalizedProbabilityPredictorMixin":

        # Functions:

        IMarginalizedProbabilityPredictorConfig& useMarginalizedProbabilityPredictor()


    cdef cppclass IAutomaticProbabilityPredictorMixin\
        "boosting::IBoostingRuleLearner::IAutomaticProbabilityPredictorMixin":
        
        # Functions:

        void useAutomaticProbabilityPredictor()
