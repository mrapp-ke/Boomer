#include "boosting/rule_evaluation/head_type_auto.hpp"
#include "boosting/rule_evaluation/head_type_complete.hpp"
#include "boosting/rule_evaluation/head_type_single.hpp"


namespace boosting {

    AutomaticHeadConfig::AutomaticHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
                                             const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr,
                                             const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
                                             const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : labelBinningConfigPtr_(labelBinningConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr),
          l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix,
            const ILabelWiseLossConfig& lossConfig) const {
        SingleLabelHeadConfig headConfig(labelBinningConfigPtr_, multiThreadingConfigPtr_, l1RegularizationConfigPtr_,
                                         l2RegularizationConfigPtr_);
        return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig);
    }

    std::unique_ptr<IStatisticsProviderFactory> AutomaticHeadConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix,
            const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        CompleteHeadConfig headConfig(labelBinningConfigPtr_, multiThreadingConfigPtr_, l1RegularizationConfigPtr_,
                                      l2RegularizationConfigPtr_);
        return headConfig.createStatisticsProviderFactory(featureMatrix, labelMatrix, lossConfig, blas, lapack);
    }

}
