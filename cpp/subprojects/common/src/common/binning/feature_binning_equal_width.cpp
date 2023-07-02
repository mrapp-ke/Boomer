#include "common/binning/feature_binning_equal_width.hpp"

#include "common/binning/bin_index_vector_dense.hpp"
#include "common/binning/bin_index_vector_dok.hpp"
#include "common/math/math.hpp"
#include "common/thresholds/thresholds_approximate.hpp"
#include "common/util/validation.hpp"
#include "feature_binning_nominal.hpp"

#include <tuple>
#include <unordered_set>

static inline std::tuple<uint32, float32, float32> preprocess(const FeatureVector& featureVector, bool sparse,
                                                              float32 binRatio, uint32 minBins, uint32 maxBins) {
    std::tuple<uint32, float32, float32> result;
    uint32 numElements = featureVector.getNumElements();

    if (numElements > 0) {
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        float32 minValue;
        uint32 i;

        if (sparse) {
            minValue = 0;
            i = 0;
        } else {
            minValue = featureIterator[0].value;
            i = 1;
        }

        float32 maxValue = minValue;
        uint32 numDistinctValues = 1;
        std::unordered_set<float32> distinctValues;

        for (; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if ((!sparse || currentValue != 0) && distinctValues.insert(currentValue).second) {
                numDistinctValues++;

                if (currentValue < minValue) {
                    minValue = currentValue;
                }

                if (currentValue > maxValue) {
                    maxValue = currentValue;
                }
            }
        }

        std::get<0>(result) =
          numDistinctValues > 1 ? calculateBoundedFraction(numDistinctValues, binRatio, minBins, maxBins) : 0;
        std::get<1>(result) = minValue;
        std::get<2>(result) = maxValue;
    } else {
        std::get<0>(result) = 0;
    }

    return result;
}

static inline uint32 getBinIndex(float32 value, float32 min, float32 width, uint32 numBins) {
    uint32 binIndex = (uint32) std::floor((value - min) / width);
    return binIndex >= numBins ? numBins - 1 : binIndex;
}

/**
 * An implementation of the type `IFeatureBinning` that assigns numerical feature values to bins, such that each bin
 * contains values from equally sized value ranges.
 */
class EqualWidthFeatureBinning final : public IFeatureBinning {
    private:

        const float32 binRatio_;

        const uint32 minBins_;

        const uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualWidthFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
            : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

        Result createBins(FeatureVector& featureVector, uint32 numExamples) const override {
            Result result;
            uint32 numElements = featureVector.getNumElements();
            bool sparse = numElements < numExamples;
            std::tuple<uint32, float32, float32> tuple =
              preprocess(featureVector, sparse, binRatio_, minBins_, maxBins_);
            uint32 numBins = std::get<0>(tuple);
            result.thresholdVectorPtr = std::make_unique<ThresholdVector>(featureVector, numBins, true);

            if (sparse) {
                result.binIndicesPtr = std::make_unique<DokBinIndexVector>();
            } else {
                result.binIndicesPtr = std::make_unique<DenseBinIndexVector>(numElements);
            }

            if (numBins > 0) {
                IBinIndexVector& binIndices = *result.binIndicesPtr;
                ThresholdVector& thresholdVector = *result.thresholdVectorPtr;
                FeatureVector::const_iterator featureIterator = featureVector.cbegin();
                ThresholdVector::iterator thresholdIterator = thresholdVector.begin();
                float32 min = std::get<1>(tuple);
                float32 max = std::get<2>(tuple);
                float32 width = (max - min) / numBins;
                uint32 sparseBinIndex;

                // If there are any sparse values, identify the bin they belong to...
                if (sparse) {
                    sparseBinIndex = getBinIndex(0, min, width, numBins);
                    thresholdIterator[sparseBinIndex] = 1;
                    thresholdVector.setSparseBinIndex(sparseBinIndex);
                } else {
                    sparseBinIndex = numBins;
                }

                // Iterate all non-sparse feature values and identify the bins they belong to...
                for (uint32 i = 0; i < numElements; i++) {
                    float32 currentValue = featureIterator[i].value;

                    if (!sparse || currentValue != 0) {
                        uint32 binIndex = getBinIndex(currentValue, min, width, numBins);

                        if (binIndex != sparseBinIndex) {
                            thresholdIterator[binIndex] = 1;
                            binIndices.setBinIndex(featureIterator[i].index, binIndex);
                        }
                    }
                }

                // Remove empty bins and calculate thresholds...
                uint32* mapping = new uint32[numBins];
                uint32 n = 0;

                for (uint32 i = 0; i < numBins; i++) {
                    mapping[i] = n;

                    if (thresholdIterator[i] > 0) {
                        thresholdIterator[n] = min + ((i + 1) * width);
                        n++;
                    }
                }

                thresholdVector.setNumElements(n, true);

                // Adjust bin indices...
                DokBinIndexVector* dokBinIndices = dynamic_cast<DokBinIndexVector*>(&binIndices);

                if (dokBinIndices) {
                    for (auto it = dokBinIndices->begin(); it != dokBinIndices->end(); it++) {
                        uint32 binIndex = it->second;
                        it->second = mapping[binIndex];
                    }
                } else {
                    for (uint32 i = 0; i < numElements; i++) {
                        uint32 binIndex = binIndices.getBinIndex(i);
                        binIndices.setBinIndex(i, mapping[binIndex]);
                    }
                }

                delete[] mapping;
            }

            return result;
        }
};

/**
 * Allows to create instances of the type `IFeatureBinning` that assign numerical feature values to bins, such that each
 * bin contains values from equally sized value ranges.
 */
class EqualWidthFeatureBinningFactory final : public IFeatureBinningFactory {
    private:

        const float32 binRatio_;

        const uint32 minBins_;

        const uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualWidthFeatureBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins)
            : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

        std::unique_ptr<IFeatureBinning> create() const override {
            return std::make_unique<EqualWidthFeatureBinning>(binRatio_, minBins_, maxBins_);
        }
};

EqualWidthFeatureBinningConfig::EqualWidthFeatureBinningConfig(
  const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : binRatio_(0.33f), minBins_(2), maxBins_(0), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

float32 EqualWidthFeatureBinningConfig::getBinRatio() const {
    return binRatio_;
}

IEqualWidthFeatureBinningConfig& EqualWidthFeatureBinningConfig::setBinRatio(float32 binRatio) {
    assertGreater<float32>("binRatio", binRatio, 0);
    assertLess<float32>("binRatio", binRatio, 1);
    binRatio_ = binRatio;
    return *this;
}

uint32 EqualWidthFeatureBinningConfig::getMinBins() const {
    return minBins_;
}

IEqualWidthFeatureBinningConfig& EqualWidthFeatureBinningConfig::setMinBins(uint32 minBins) {
    assertGreaterOrEqual<uint32>("minBins", minBins, 2);
    minBins_ = minBins;
    return *this;
}

uint32 EqualWidthFeatureBinningConfig::getMaxBins() const {
    return maxBins_;
}

IEqualWidthFeatureBinningConfig& EqualWidthFeatureBinningConfig::setMaxBins(uint32 maxBins) {
    if (maxBins != 0) assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins_);
    maxBins_ = maxBins;
    return *this;
}

std::unique_ptr<IThresholdsFactory> EqualWidthFeatureBinningConfig::createThresholdsFactory(
  const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    std::unique_ptr<IFeatureBinningFactory> numericalFeatureBinningFactoryPtr =
      std::make_unique<EqualWidthFeatureBinningFactory>(binRatio_, minBins_, maxBins_);
    std::unique_ptr<IFeatureBinningFactory> nominalFeatureBinningFactoryPtr =
      std::make_unique<NominalFeatureBinningFactory>();
    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
    return std::make_unique<ApproximateThresholdsFactory>(std::move(numericalFeatureBinningFactoryPtr),
                                                          std::move(nominalFeatureBinningFactoryPtr), numThreads);
}
