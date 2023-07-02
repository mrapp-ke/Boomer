#include "boosting/statistics/statistic_format_sparse.hpp"

namespace boosting {

    SparseStatisticsConfig::SparseStatisticsConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {}

    std::unique_ptr<IStatisticsProviderFactory> SparseStatisticsConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack) const {
        return lossConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, blas, lapack, true);
    }

    bool SparseStatisticsConfig::isDense() const {
        return false;
    }

    bool SparseStatisticsConfig::isSparse() const {
        return true;
    }

}
