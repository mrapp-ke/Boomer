#include "boosting/learner_boomer.hpp"

namespace boosting {

    Boomer::Config::Config() {
        this->useSequentialRuleModelAssemblage();
        this->useGreedyTopDownRuleInduction();
        this->useDefaultRule();
        this->useNoLabelSampling();
        this->useNoInstanceSampling();
        this->useFeatureSamplingWithoutReplacement();
        this->useParallelPrediction();
        this->useAutomaticDefaultRule();
        this->useAutomaticPartitionSampling();
        this->useAutomaticFeatureBinning();
        this->useSizeStoppingCriterion();
        this->useNoTimeStoppingCriterion();
        this->useNoRulePruning();
        this->useNoGlobalPruning();
        this->useNoSequentialPostOptimization();
        this->useConstantShrinkagePostProcessor();
        this->useAutomaticParallelRuleRefinement();
        this->useAutomaticParallelStatisticUpdate();
        this->useAutomaticHeads();
        this->useAutomaticStatistics();
        this->useLabelWiseLogisticLoss();
        this->useNoL1Regularization();
        this->useL2Regularization();
        this->useAutomaticLabelBinning();
        this->useAutomaticBinaryPredictor();
        this->useLabelWiseScorePredictor();
        this->useAutomaticProbabilityPredictor();
    }

    ISizeStoppingCriterionConfig& Boomer::Config::useSizeStoppingCriterion() {
        ISizeStoppingCriterionConfig& ref = ISizeStoppingCriterionMixin::useSizeStoppingCriterion();
        ref.setMaxRules(1000);
        return ref;
    }

    Boomer::Boomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                   Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction)
        : AbstractBoostingRuleLearner(*configPtr, ddotFunction, dspmvFunction, dsysvFunction),
          configPtr_(std::move(configPtr)) {}

    std::unique_ptr<IBoomer::IConfig> createBoomerConfig() {
        return std::make_unique<Boomer::Config>();
    }

    std::unique_ptr<IBoomer> createBoomer(std::unique_ptr<IBoomer::IConfig> configPtr, Blas::DdotFunction ddotFunction,
                                          Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction) {
        return std::make_unique<Boomer>(std::move(configPtr), ddotFunction, dspmvFunction, dsysvFunction);
    }

}
