#include "common/binning/feature_binning_no.hpp"
#include "common/thresholds/thresholds_exact.hpp"


NoFeatureBinningConfig::NoFeatureBinningConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : multiThreadingConfigPtr_(multiThreadingConfigPtr) {

}

std::unique_ptr<IThresholdsFactory> NoFeatureBinningConfig::createThresholdsFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
    return std::make_unique<ExactThresholdsFactory>(numThreads);
}
