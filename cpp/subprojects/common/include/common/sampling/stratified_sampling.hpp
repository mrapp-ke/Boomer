/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include "common/input/label_matrix_csc.hpp"
#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/random.hpp"
#include <unordered_map>
#include <set>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>


static inline void updateNumExamplesPerLabel(const CContiguousLabelMatrix& labelMatrix, uint32 exampleIndex,
                                             uint32* numExamplesPerLabel,
                                             std::unordered_map<uint32, uint32>& affectedLabelIndices) {
    CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.getNumCols();

    for (uint32 i = 0; i < numLabels; i++) {
        if (labelIterator[i]) {
            uint32 numRemaining = numExamplesPerLabel[i];
            numExamplesPerLabel[i] = numRemaining - 1;
            affectedLabelIndices.emplace(i, numRemaining);
        }
    }
}

static inline void updateNumExamplesPerLabel(const CsrLabelMatrix& labelMatrix, uint32 exampleIndex,
                                             uint32* numExamplesPerLabel,
                                             std::unordered_map<uint32, uint32>& affectedLabelIndices) {
    CsrLabelMatrix::index_const_iterator indexIterator = labelMatrix.row_indices_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.row_indices_cend(exampleIndex) - indexIterator;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indexIterator[i];
        uint32 numRemaining = numExamplesPerLabel[labelIndex];
        numExamplesPerLabel[labelIndex] = numRemaining - 1;
        affectedLabelIndices.emplace(labelIndex, numRemaining);
    }
}

static inline bool tiebreak(uint32 numDesiredSamples, uint32 numDesiredOutOfSamples, RNG& rng) {
    if (numDesiredSamples > numDesiredOutOfSamples) {
        return true;
    } else if (numDesiredSamples < numDesiredOutOfSamples) {
        return false;
    } else {
        return rng.random(0, 2) != 0;
    }
}

/**
 * Implements stratified sampling, where distinct label vectors are treated as individual classes.
 *
 * @tparam LabelMatrix      The type of the label matrix that provides random or row-wise access to the labels of the
 *                          training examples
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of the examples that should be
 *                          considered
 */
template<typename LabelMatrix, typename IndexIterator>
class ExampleWiseStratification final {

    private:

        uint32 numTotal_;

        typedef typename LabelMatrix::view_type Key;

        typedef typename LabelMatrix::view_type::Hash Hash;

        typedef typename LabelMatrix::view_type::Pred Pred;

        std::unordered_map<Key, std::vector<uint32>, Hash, Pred> map_;

        std::vector<std::reference_wrapper<std::vector<uint32>>> order_;

    public:

        /**
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param indicesBegin  An iterator to the beginning of the indices of the examples that should be considered
         * @param indicesEnd    An iterator to the end of the indices of hte examples that should be considered
         */
        ExampleWiseStratification(const LabelMatrix& labelMatrix, IndexIterator indicesBegin,
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

        /**
         * Randomly selects a stratified sample of the available examples and sets their weights to 1, while the
         * remaining weights are set to 0.
         *
         * @param weightVector  A reference to an object of type `BitWeightVector`, the weights should be written to
         * @param sampleSize    The fraction of the available examples to be selected
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         */
        void sampleWeights(BitWeightVector& weightVector, float32 sampleSize, RNG& rng) const {
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
                uint32 numSamples = (uint32) (tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ?
                                              std::ceil(numSamplesDecimal) : std::floor(numSamplesDecimal));
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

        /**
         * Randomly splits the available examples into two distinct sets and updates a given `BiPartition` accordingly.
         *
         * @param partition A reference to an object of type `BiPartition` to be updated
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         */
        void sampleBiPartition(BiPartition& partition, RNG& rng) const {
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
                uint32 numSamples = (uint32) (tiebreak(numFirst, numSecond, rng) ? std::ceil(numSamplesDecimal)
                                                                                 : std::floor(numSamplesDecimal));
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

};

/**
 * Implements iterative stratified sampling for selecting a subset of the available training examples as proposed in the
 * following publication:
 *
 * Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-label Data. In: Machine Learning and
 * Knowledge Discovery in Databases. ECML PKDD 2011. Lecture Notes in Computer Science, vol 6913. Springer.
 *
 * @tparam LabelMatrix      The type of the label matrix that provides random or row-wise access to the labels of the
 *                          training examples
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of the examples that should be
 *                          considered
 */
template<typename LabelMatrix, typename IndexIterator>
class LabelWiseStratification final {

    private:

        uint32 numRows_;

        uint32 numCols_;

        uint32* rowIndices_;

        uint32* colIndices_;

    public:

        /**
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param indicesBegin  An iterator to the beginning of the indices of the examples that should be considered
         * @param indicesEnd    An iterator to the end of the indices of the examples that should be considered
         */
        LabelWiseStratification(const LabelMatrix& labelMatrix, IndexIterator indicesBegin, IndexIterator indicesEnd)
            : numRows_(indicesEnd - indicesBegin) {
            // Convert the given label matrix into the CSC format...
            const CscLabelMatrix cscLabelMatrix(labelMatrix, indicesBegin, indicesEnd);

            // Create an array that stores for each label the number of examples that are associated with the label, as
            // well as a sorted map that stores all label indices in increasing order of the number of associated
            // examples...
            uint32 numLabels = cscLabelMatrix.getNumCols();
            uint32 numExamplesPerLabel[numLabels];
            typedef std::set<IndexedValue<uint32>, IndexedValue<uint32>::Compare> SortedSet;
            SortedSet sortedLabelIndices;

            for (uint32 i = 0; i < numLabels; i++) {
                uint32 numExamples = cscLabelMatrix.column_indices_cend(i) - cscLabelMatrix.column_indices_cbegin(i);
                numExamplesPerLabel[i] = numExamples;

                if (numExamples > 0) {
                    sortedLabelIndices.emplace(i, numExamples);
                }
            }

            // Allocate arrays for storing the row and column indices of the labels to be processed by the sampling
            // method in the CSC format...
            rowIndices_ = (uint32*) malloc(cscLabelMatrix.getNumNonZeroElements() * sizeof(uint32));
            colIndices_ = (uint32*) malloc((sortedLabelIndices.size() + 1) * sizeof(uint32));
            uint32 numNonZeroElements = 0;
            uint32 numCols = 0;

            // Create a boolean array that stores whether individual examples remain to be processed (1) or not (0)...
            uint32 numTotalExamples = labelMatrix.getNumRows();
            uint8 mask[numTotalExamples] = {};

            for (uint32 i = 0; i < numRows_; i++) {
                uint32 exampleIndex = indicesBegin[i];
                mask[exampleIndex] = 1;
            }

            // As long as there are labels that have not been processed yet, proceed with the label that has the
            // smallest number of associated examples...
            std::unordered_map<uint32, uint32> affectedLabelIndices;
            SortedSet::iterator firstEntry;

            while ((firstEntry = sortedLabelIndices.begin()) != sortedLabelIndices.end()) {
                const IndexedValue<uint32>& entry = *firstEntry;
                uint32 labelIndex = entry.index;

                // Remove the label from the sorted map...
                sortedLabelIndices.erase(firstEntry);

                // Add the number of non-zero labels that have been processed so far to the array of column indices...
                colIndices_[numCols] = numNonZeroElements;
                numCols++;

                // Iterate the examples that are associated with the current label, if no weight has been set yet...
                CscLabelMatrix::index_const_iterator indexIterator = cscLabelMatrix.column_indices_cbegin(labelIndex);
                uint32 numExamples = cscLabelMatrix.column_indices_cend(labelIndex) - indexIterator;

                for (uint32 i = 0; i < numExamples; i++) {
                    uint32 exampleIndex = indexIterator[i];

                    // If the example has not been processed yet...
                    if (mask[exampleIndex]) {
                        mask[exampleIndex] = 0;

                        // Add the example's index to the array of row indices...
                        rowIndices_[numNonZeroElements] = exampleIndex;
                        numNonZeroElements++;

                        // For each label that is associated with the example, decrement the number of associated
                        // examples by one...
                        updateNumExamplesPerLabel(labelMatrix, exampleIndex, &numExamplesPerLabel[0],
                                                  affectedLabelIndices);
                    }
                }

                // Remove each label, for which the number of associated examples have been changed previously, from the
                // sorted map and add it again to update the order...
                for (auto it = affectedLabelIndices.cbegin(); it != affectedLabelIndices.cend(); it++) {
                    uint32 key = it->first;

                    if (key != labelIndex) {
                        uint32 value = it->second;
                        SortedSet::iterator it2 = sortedLabelIndices.find(IndexedValue<uint32>(key, value));
                        uint32 numRemaining = numExamplesPerLabel[key];

                        if (numRemaining > 0) {
                            sortedLabelIndices.emplace_hint(it2, key, numRemaining);
                        }

                        sortedLabelIndices.erase(it2);
                    }
                }

                affectedLabelIndices.clear();
            }

            // If there are examples that are not associated with any labels, we handle them separately..
            uint32 numRemaining = numRows_ - numNonZeroElements;

            if (numRemaining > 0) {
                // Adjust the size of the arrays that are used to store row and column indices...
                rowIndices_ = (uint32*) realloc(rowIndices_, (numNonZeroElements + numRemaining) * sizeof(uint32));
                colIndices_ = (uint32*) realloc(colIndices_, (numCols + 1) * sizeof(uint32));

                // Add the number of non-zero labels that have been processed so far to the array of column indices...
                colIndices_[numCols] = numNonZeroElements;
                numCols++;

                // Iterate the weights of all examples to find those whose weight has not been set yet...
                for (uint32 i = 0; i < numTotalExamples; i++) {
                    if (mask[i]) {
                        // Add the example's index to the array of row indices...
                        rowIndices_[numNonZeroElements] = i;
                        numNonZeroElements++;
                    }
                }
            } else {
                // Adjust the size of the arrays that are used to store row and column indices...
                rowIndices_ = (uint32*) realloc(rowIndices_, numNonZeroElements * sizeof(uint32));
                colIndices_ = (uint32*) realloc(colIndices_, numCols * sizeof(uint32));
            }

            colIndices_[numCols - 1] = numNonZeroElements;
            numCols_ = numCols - 1;
        }

        /**
         * Randomly selects a stratified sample of the available examples and sets their weights to 1, while the
         * remaining weights are set to 0.
         *
         * @param weightVector  A reference to an object of type `BitWeightVector`, the weights should be written to
         * @param sampleSize    The fraction of the available examples to be selected
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         */
        void sampleWeights(BitWeightVector& weightVector, float32 sampleSize, RNG& rng) const {
            uint32 numTotalSamples = (uint32) std::round(sampleSize * numRows_);
            uint32 numTotalOutOfSamples = numRows_ - numTotalSamples;
            uint32 numNonZeroWeights = 0;
            uint32 numZeroWeights = 0;

            // For each column, assign a weight to the corresponding examples...
            for (uint32 i = 0; i < numCols_; i++) {
                uint32 start = colIndices_[i];
                uint32* exampleIndices = &rowIndices_[start];
                uint32 end = colIndices_[i + 1];
                uint32 numExamples = end - start;
                float32 numSamplesDecimal = sampleSize * numExamples;
                uint32 numDesiredSamples = numTotalSamples - numNonZeroWeights;
                uint32 numDesiredOutOfSamples = numTotalOutOfSamples - numZeroWeights;
                uint32 numSamples = (uint32) (tiebreak(numDesiredSamples, numDesiredOutOfSamples, rng) ?
                                              std::ceil(numSamplesDecimal) : std::floor(numSamplesDecimal));
                numNonZeroWeights += numSamples;
                numZeroWeights += (numExamples - numSamples);
                uint32 j;

                // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and set their weights to 1...
                for (j = 0; j < numSamples; j++) {
                    uint32 randomIndex = rng.random(j, numExamples);
                    uint32 exampleIndex = exampleIndices[randomIndex];
                    exampleIndices[randomIndex] = exampleIndices[j];
                    exampleIndices[j] = exampleIndex;
                    weightVector.set(exampleIndex, true);
                }

                // Set the weights of the remaining examples to 0...
                for (; j < numExamples; j++) {
                    uint32 exampleIndex = exampleIndices[j];
                    weightVector.set(exampleIndex, false);
                }
            }

            weightVector.setNumNonZeroWeights(numNonZeroWeights);
        }

        /**
         * Randomly splits the available examples into two distinct sets and updates a given `BiPartition` accordingly.
         *
         * @param partition A reference to an object of type `BiPartition` to be updated
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         */
        void sampleBiPartition(BiPartition& partition, RNG& rng) const {
            BiPartition::iterator firstIterator = partition.first_begin();
            BiPartition::iterator secondIterator = partition.second_begin();
            uint32 numFirst = partition.getNumFirst();
            uint32 numSecond = partition.getNumSecond();

            for (uint32 i = 0; i < numCols_; i++) {
                uint32 start = colIndices_[i];
                uint32* exampleIndices = &rowIndices_[start];
                uint32 end = colIndices_[i + 1];
                uint32 numExamples = end - start;

                float32 sampleSize = (float32) numFirst / (float32) (numFirst + numSecond);
                float32 numSamplesDecimal = sampleSize * numExamples;
                uint32 numSamples = (uint32) (tiebreak(numFirst, numSecond, rng) ? std::ceil(numSamplesDecimal)
                                                                                 : std::floor(numSamplesDecimal));
                numFirst -= numSamples;
                numSecond -= (numExamples - numSamples);

                // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples and add them to the first set...
                uint32 j;

                for (j = 0; j < numSamples; j++) {
                    uint32 randomIndex = rng.random(j, numExamples);
                    uint32 exampleIndex = exampleIndices[randomIndex];
                    exampleIndices[randomIndex] = exampleIndices[j];
                    exampleIndices[j] = exampleIndex;
                    *firstIterator = exampleIndex;
                    firstIterator++;
                }

                // Add the remaining examples to the second set...
                for (; j < numExamples; j++) {
                    uint32 exampleIndex = exampleIndices[j];
                    *secondIterator = exampleIndex;
                    secondIterator++;
                }
            }
        }

};
