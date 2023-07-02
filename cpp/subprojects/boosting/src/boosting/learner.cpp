#include "boosting/learner.hpp"

#include "boosting/model/rule_list_builder.hpp"
#include "boosting/rule_evaluation/rule_compare_function.hpp"

namespace boosting {

    AbstractBoostingRuleLearner::Config::Config()
        : AbstractRuleLearner::Config(BOOSTED_RULE_COMPARE_FUNCTION),
          headConfigPtr_(std::make_unique<CompleteHeadConfig>(labelBinningConfigPtr_, parallelStatisticUpdateConfigPtr_,
                                                              l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)),
          statisticsConfigPtr_(std::make_unique<DenseStatisticsConfig>(lossConfigPtr_)),
          lossConfigPtr_(std::make_unique<LabelWiseLogisticLossConfig>(headConfigPtr_)),
          l1RegularizationConfigPtr_(std::make_unique<NoRegularizationConfig>()),
          l2RegularizationConfigPtr_(std::make_unique<NoRegularizationConfig>()),
          labelBinningConfigPtr_(
            std::make_unique<NoLabelBinningConfig>(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)) {}

    std::unique_ptr<IHeadConfig>& AbstractBoostingRuleLearner::Config::getHeadConfigPtr() {
        return headConfigPtr_;
    }

    std::unique_ptr<IStatisticsConfig>& AbstractBoostingRuleLearner::Config::getStatisticsConfigPtr() {
        return statisticsConfigPtr_;
    }

    std::unique_ptr<IRegularizationConfig>& AbstractBoostingRuleLearner::Config::getL1RegularizationConfigPtr() {
        return l1RegularizationConfigPtr_;
    }

    std::unique_ptr<IRegularizationConfig>& AbstractBoostingRuleLearner::Config::getL2RegularizationConfigPtr() {
        return l2RegularizationConfigPtr_;
    }

    std::unique_ptr<ILossConfig>& AbstractBoostingRuleLearner::Config::getLossConfigPtr() {
        return lossConfigPtr_;
    }

    std::unique_ptr<ILabelBinningConfig>& AbstractBoostingRuleLearner::Config::getLabelBinningConfigPtr() {
        return labelBinningConfigPtr_;
    }

    AbstractBoostingRuleLearner::AbstractBoostingRuleLearner(IBoostingRuleLearner::IConfig& config,
                                                             Blas::DdotFunction ddotFunction,
                                                             Blas::DspmvFunction dspmvFunction,
                                                             Lapack::DsysvFunction dsysvFunction)
        : AbstractRuleLearner(config), config_(config), blas_(Blas(ddotFunction, dspmvFunction)),
          lapack_(Lapack(dsysvFunction)) {}

    std::unique_ptr<IStatisticsProviderFactory> AbstractBoostingRuleLearner::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const {
        return config_.getStatisticsConfigPtr()->createStatisticsProviderFactory(featureMatrix, labelMatrix, blas_,
                                                                                 lapack_);
    }

    std::unique_ptr<IModelBuilderFactory> AbstractBoostingRuleLearner::createModelBuilderFactory() const {
        return std::make_unique<RuleListBuilderFactory>();
    }

}
