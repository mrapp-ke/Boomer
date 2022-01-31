#include "common/binning/feature_binning_equal_frequency.hpp"
#include "common/binning/bin_index_vector_dense.hpp"
#include "common/binning/bin_index_vector_dok.hpp"
#include "common/binning/binning.hpp"
#include "common/math/math.hpp"
#include "common/thresholds/thresholds_approximate.hpp"
#include "common/util/validation.hpp"
#include "feature_binning_nominal.hpp"


static inline uint32 getNumBins(FeatureVector& featureVector, bool sparse, float32 binRatio, uint32 minBins,
                                uint32 maxBins) {
    uint32 numElements = featureVector.getNumElements();

    if (numElements > 0) {
        featureVector.sortByValues();
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        uint32 numDistinctValues = 1;
        float32 previousValue;
        uint32 i;

        if (sparse) {
            previousValue = 0;
            i = 0;
        } else {
            previousValue = featureIterator[0].value;
            i = 1;
        }

        for (; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if ((!sparse || currentValue != 0) && currentValue != previousValue) {
                numDistinctValues++;
                previousValue = currentValue;
            }
        }

        return numDistinctValues > 1 ? calculateNumBins(numDistinctValues, binRatio, minBins, maxBins) : 0;
    }

    return 0;
}

/**
 * An implementation of the type `IFeatureBinning` that assigns numerical feature values to bins, such that each bin
 * contains approximately the same number of values.
 */
class EqualFrequencyFeatureBinning final : public IFeatureBinning {

    private:

        float32 binRatio_;

        uint32 minBins_;

        uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualFrequencyFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
            : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

        }

        Result createBins(FeatureVector& featureVector, uint32 numExamples) const override {
            Result result;
            uint32 numElements = featureVector.getNumElements();
            uint32 numSparse = numExamples - numElements;
            bool sparse = numSparse > 0;
            uint32 numBins = getNumBins(featureVector, sparse, binRatio_, minBins_, maxBins_);
            result.thresholdVectorPtr = std::make_unique<ThresholdVector>(featureVector, numBins);

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
                uint32 numElementsPerBin = (uint32) std::ceil((float) numElements / (float) numBins);
                uint32 binIndex = 0;
                float32 previousValue = 0;
                uint32 i = 0;

                // Iterate feature values < 0...
                for (; i < numElements; i++) {
                    float32 currentValue = featureIterator[i].value;

                    if (currentValue >= 0) {
                        break;
                    }

                    if (currentValue != previousValue) {
                        if (i / numElementsPerBin != binIndex) {
                            thresholdIterator[binIndex] = arithmeticMean(previousValue, currentValue);
                            binIndex++;
                        }

                        previousValue = currentValue;
                    }

                    binIndices.setBinIndex(featureIterator[i].index, binIndex);
                }

                // If there are any sparse values, check if they belong to the current one or the next one...
                if (sparse) {
                    if (i / numElementsPerBin != binIndex) {
                        thresholdIterator[binIndex] = arithmeticMean<float32>(previousValue, 0);
                        binIndex++;
                    }

                    thresholdVector.setSparseBinIndex(binIndex);
                    previousValue = 0;
                }

                // Iterate feature values >= 0...
                for (; i < numElements; i++) {
                    float32 currentValue = featureIterator[i].value;

                    if (!sparse || currentValue != 0) {
                        if (currentValue != previousValue) {
                            if ((i + numSparse) / numElementsPerBin != binIndex) {
                                thresholdIterator[binIndex] = arithmeticMean(previousValue, currentValue);
                                binIndex++;
                            }

                            previousValue = currentValue;
                        }

                        binIndices.setBinIndex(featureIterator[i].index, binIndex);
                    }
                }

                thresholdVector.setNumElements(binIndex + 1, true);
            }

            return result;
        }

};

/**
 * Allows to create instances of the type `IFeatureBinning` that assign numerical feature values to bins, such that each
 * bin contains approximately the same number of values.
 */
class EqualFrequencyFeatureBinningFactory final : public IFeatureBinningFactory {

    private:

        float32 binRatio_;

        uint32 minBins_;

        uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualFrequencyFeatureBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins)
            : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

        }

        std::unique_ptr<IFeatureBinning> create() const override {
            return std::make_unique<EqualFrequencyFeatureBinning>(binRatio_, minBins_, maxBins_);
        }

};

EqualFrequencyFeatureBinningConfig::EqualFrequencyFeatureBinningConfig(
        const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : binRatio_(0.33f), minBins_(2), maxBins_(0), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

}

float32 EqualFrequencyFeatureBinningConfig::getBinRatio() const {
    return binRatio_;
}

IEqualFrequencyFeatureBinningConfig& EqualFrequencyFeatureBinningConfig::setBinRatio(float32 binRatio) {
    assertGreater<float32>("binRatio", binRatio, 0);
    assertLess<float32>("binRatio", binRatio, 1);
    binRatio_ = binRatio;
    return *this;
}

uint32 EqualFrequencyFeatureBinningConfig::getMinBins() const {
    return minBins_;
}

IEqualFrequencyFeatureBinningConfig& EqualFrequencyFeatureBinningConfig::setMinBins(uint32 minBins) {
    assertGreaterOrEqual<uint32>("minBins", minBins, 2);
    minBins_ = minBins;
    return *this;
}

uint32 EqualFrequencyFeatureBinningConfig::getMaxBins() const {
    return maxBins_;
}

IEqualFrequencyFeatureBinningConfig& EqualFrequencyFeatureBinningConfig::setMaxBins(uint32 maxBins) {
    if (maxBins != 0) { assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins_); }
    maxBins_ = maxBins;
    return *this;
}

std::unique_ptr<IThresholdsFactory> EqualFrequencyFeatureBinningConfig::createThresholdsFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    std::unique_ptr<IFeatureBinningFactory> numericalFeatureBinningFactoryPtr =
        std::make_unique<EqualFrequencyFeatureBinningFactory>(binRatio_, minBins_, maxBins_);
    std::unique_ptr<IFeatureBinningFactory> nominalFeatureBinningFactoryPtr =
        std::make_unique<NominalFeatureBinningFactory>();
    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
    return std::make_unique<ApproximateThresholdsFactory>(std::move(numericalFeatureBinningFactoryPtr),
                                                          std::move(nominalFeatureBinningFactoryPtr), numThreads);
}
