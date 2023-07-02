#include "common/sampling/stratified_sampling_example_wise.hpp"

#include "common/sampling/partition_single.hpp"
#include "stratified_sampling_common.hpp"

#include <algorithm>

template<typename LabelMatrix, typename IndexIterator>
ExampleWiseStratification<LabelMatrix, IndexIterator>::ExampleWiseStratification(const LabelMatrix& labelMatrix,
                                                                                 IndexIterator indicesBegin,
                                                                                 IndexIterator indicesEnd)
    : numTotal_(indicesEnd - indicesBegin) {
    // Create a map that stores the indices of the examples that are associated with each unique label vector...
    for (uint32 i = 0; i < numTotal_; i++) {
        uint32 exampleIndex = indicesBegin[i];
        std::vector<uint32>& exampleIndices = map_[labelMatrix.createView(exampleIndex)];
        exampleIndices.push_back(exampleIndex);
    }

    // Sort the label vectors by their frequency...
    order_.reserve(map_.size());

    for (auto it = map_.begin(); it != map_.end(); it++) {
        auto& entry = *it;
        std::vector<uint32>& exampleIndices = entry.second;
        order_.push_back(exampleIndices);
    }

    std::sort(order_.begin(), order_.end(), [=](const std::vector<uint32>& a, const std::vector<uint32>& b) {
        return a.size() < b.size();
    });
}

template<typename LabelMatrix, typename IndexIterator>
void ExampleWiseStratification<LabelMatrix, IndexIterator>::sampleWeights(BitWeightVector& weightVector,
                                                                          float32 sampleSize, RNG& rng) const {
    uint32 numTotalSamples = (uint32) std::round(sampleSize * numTotal_);
    uint32 numTotalOutOfSamples = numTotal_ - numTotalSamples;
    uint32 numNonZeroWeights = 0;
    uint32 numZeroWeights = 0;

    for (auto it = order_.begin(); it != order_.end(); it++) {
        std::vector<uint32>& exampleIndices = *it;
        std::vector<uint32>::iterator indexIterator = exampleIndices.begin();
        uint32 numExamples = exampleIndices.size();
        float32 numSamplesDecimal = sampleSize * numExamples;
        uint32 numDesiredSamples = numTotalSamples - numNonZeroWeights;
        uint32 numDesiredOutOfSamples = numTotalOutOfSamples - numZeroWeights;
        uint32 numSamples =
          (uint32) (tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ? std::ceil(numSamplesDecimal)
                                                                             : std::floor(numSamplesDecimal));
        numNonZeroWeights += numSamples;
        numZeroWeights += (numExamples - numSamples);

        // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and set their weight to 1...
        uint32 i;

        for (i = 0; i < numSamples; i++) {
            uint32 randomIndex = rng.random(i, numExamples);
            uint32 exampleIndex = indexIterator[randomIndex];
            indexIterator[randomIndex] = indexIterator[i];
            indexIterator[i] = exampleIndex;
            weightVector.set(exampleIndex, true);
        }

        // Set the weights of the remaining examples to 0...
        for (; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            weightVector.set(exampleIndex, false);
        }
    }

    weightVector.setNumNonZeroWeights(numNonZeroWeights);
}

template<typename LabelMatrix, typename IndexIterator>
void ExampleWiseStratification<LabelMatrix, IndexIterator>::sampleBiPartition(BiPartition& partition, RNG& rng) const {
    BiPartition::iterator firstIterator = partition.first_begin();
    BiPartition::iterator secondIterator = partition.second_begin();
    uint32 numFirst = partition.getNumFirst();
    uint32 numSecond = partition.getNumSecond();

    for (auto it = order_.begin(); it != order_.end(); it++) {
        std::vector<uint32>& exampleIndices = *it;
        std::vector<uint32>::iterator indexIterator = exampleIndices.begin();
        uint32 numExamples = exampleIndices.size();
        float32 sampleSize = (float32) numFirst / (float32) (numFirst + numSecond);
        float32 numSamplesDecimal = sampleSize * numExamples;
        uint32 numSamples =
          (uint32) (tiebreak(numFirst, numSecond, rng) ? std::ceil(numSamplesDecimal) : std::floor(numSamplesDecimal));

        // Ensure that we do not add too many examples to the first or second partition...
        if (numSamples > numFirst) {
            numSamples = numFirst;
        } else if (numExamples - numSamples > numSecond) {
            numSamples = numExamples - numSecond;
        }

        numFirst -= numSamples;
        numSecond -= (numExamples - numSamples);

        // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and add them to the first set...
        uint32 i;

        for (i = 0; i < numSamples; i++) {
            uint32 randomIndex = rng.random(i, numExamples);
            uint32 exampleIndex = indexIterator[randomIndex];
            indexIterator[randomIndex] = indexIterator[i];
            indexIterator[i] = exampleIndex;
            *firstIterator = exampleIndex;
            firstIterator++;
        }

        // Add the remaining examples to the second set...
        for (; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            *secondIterator = exampleIndex;
            secondIterator++;
        }
    }
}

template class ExampleWiseStratification<CContiguousLabelMatrix, SinglePartition::const_iterator>;
template class ExampleWiseStratification<CContiguousLabelMatrix, BiPartition::const_iterator>;
template class ExampleWiseStratification<CsrLabelMatrix, SinglePartition::const_iterator>;
template class ExampleWiseStratification<CsrLabelMatrix, BiPartition::const_iterator>;
