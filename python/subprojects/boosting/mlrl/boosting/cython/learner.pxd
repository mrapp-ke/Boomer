from mlrl.common.cython.learner cimport IRuleLearner, IRuleLearnerConfig, RuleLearner, RuleLearnerConfig
from mlrl.boosting.cython.label_binning cimport IEqualWidthLabelBinningConfig
from mlrl.boosting.cython.post_processor cimport IConstantShrinkageConfig
from mlrl.boosting.cython.regularization cimport IManualRegularizationConfig

from libcpp.memory cimport unique_ptr


ctypedef double (*DdotFunction)(int* n, double* dx, int* incx, double* dy, int* incy)

ctypedef void (*DspmvFunction)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta, double* y, int* incy)

ctypedef void (*DsysvFunction)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork, int* info)


cdef extern from "boosting/learner.hpp" namespace "boosting" nogil:

    cdef cppclass IBoostingRuleLearnerConfig"boosting::IBoostingRuleLearner::IConfig"(IRuleLearnerConfig):

        # Functions:

        void useAutomaticFeatureBinning()

        IConstantShrinkageConfig& useConstantShrinkagePostProcessor()

        void useAutomaticParallelRuleRefinement()

        void useAutomaticParallelStatisticUpdate()

        void useAutomaticHeads()

        void useSingleLabelHeads()

        void useCompleteHeads()

        void useNoL1Regularization()

        IManualRegularizationConfig& useL1Regularization()

        void useNoL2Regularization()

        IManualRegularizationConfig& useL2Regularization()

        void useExampleWiseLogisticLoss()

        void useLabelWiseLogisticLoss()

        void useLabelWiseSquaredErrorLoss()

        void useLabelWiseSquaredHingeLoss()

        void useNoLabelBinning()

        void useAutomaticLabelBinning()

        IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning()

        void useExampleWiseClassificationPredictor()

        void useLabelWiseClassificationPredictor()

        void useLabelWiseRegressionPredictor()

        void useLabelWiseProbabilityPredictor()


    cdef cppclass IBoostingRuleLearner(IRuleLearner):
        pass


    unique_ptr[IBoostingRuleLearnerConfig] createBoostingRuleLearnerConfig()


    unique_ptr[IBoostingRuleLearner] createBoostingRuleLearner(unique_ptr[IBoostingRuleLearnerConfig] configPtr,
                                                               DdotFunction ddotFunction, DspmvFunction dspmvFunction,
                                                               DsysvFunction dsysvFunction)


cdef class BoostingRuleLearnerConfig(RuleLearnerConfig):

    # Attributes:

    cdef unique_ptr[IBoostingRuleLearnerConfig] rule_learner_config_ptr


cdef class BoostingRuleLearner(RuleLearner):

    # Attributes:

    cdef unique_ptr[IBoostingRuleLearner] rule_learner_ptr
