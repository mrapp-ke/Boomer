#include "common/binning/feature_binning_equal_width.hpp"
#include "common/binning/bin_index_vector_dense.hpp"
#include "common/binning/bin_index_vector_dok.hpp"
#include "common/binning/binning.hpp"
#include <unordered_set>
#include <tuple>


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
            numDistinctValues > 1 ? calculateNumBins(numDistinctValues, binRatio, minBins, maxBins) : 0;
        std::get<1>(result) = minValue;
        std::get<2>(result) = maxValue;
    } else {
        std::get<0>(result) = 0;
    }

    return result;
}

EqualWidthFeatureBinning::EqualWidthFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
    : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

}

static inline uint32 getBinIndex(float32 value, float32 min, float32 width, uint32 numBins) {
    uint32 binIndex = (uint32) std::floor((value - min) / width);
    return binIndex >= numBins ? numBins - 1 : binIndex;
}

IFeatureBinning::Result EqualWidthFeatureBinning::createBins(FeatureVector& featureVector, uint32 numExamples) const {
    Result result;
    uint32 numElements = featureVector.getNumElements();
    bool sparse = numElements < numExamples;
    std::tuple<uint32, float32, float32> tuple = preprocess(featureVector, sparse, binRatio_, minBins_, maxBins_);
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
        uint32 mapping[numBins];
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

        if (dokBinIndices != nullptr) {
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
    }

    return result;
}
