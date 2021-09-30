#include "common/binning/feature_binning_equal_frequency.hpp"
#include "common/binning/bin_index_vector_dense.hpp"
#include "common/binning/bin_index_vector_dok.hpp"
#include "common/binning/binning.hpp"
#include "common/math/math.hpp"
#include "common/validation.hpp"


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

EqualFrequencyFeatureBinning::EqualFrequencyFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
    : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {
    assertGreater<float32>("binRatio", binRatio, 0);
    assertLess<float32>("binRatio", binRatio, 1);
    assertGreaterOrEqual<uint32>("minBins", minBins, 2);
    if (maxBins != 0) { assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins); }
}

IFeatureBinning::Result EqualFrequencyFeatureBinning::createBins(FeatureVector& featureVector,
                                                                 uint32 numExamples) const {
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
