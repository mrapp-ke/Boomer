#include "boosting/statistics/statistic_format_dense.hpp"

namespace boosting {

    DenseStatisticsConfig::DenseStatisticsConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {}

    std::unique_ptr<IStatisticsProviderFactory> DenseStatisticsConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack) const {
        return lossConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack, false);
    }

    bool DenseStatisticsConfig::isDense() const {
        return true;
    }

    bool DenseStatisticsConfig::isSparse() const {
        return false;
    }

}
