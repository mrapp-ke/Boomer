/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/partition_bi.hpp"
#include "common/sampling/weight_vector_bit.hpp"

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

        const uint32 numRows_;

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
        LabelWiseStratification(const LabelMatrix& labelMatrix, IndexIterator indicesBegin, IndexIterator indicesEnd);

        ~LabelWiseStratification();

        /**
         * Randomly selects a stratified sample of the available examples and sets their weights to 1, while the
         * remaining weights are set to 0.
         *
         * @param weightVector  A reference to an object of type `BitWeightVector`, the weights should be written to
         * @param sampleSize    The fraction of the available examples to be selected
         * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be
         *                      used
         */
        void sampleWeights(BitWeightVector& weightVector, float32 sampleSize, RNG& rng) const;

        /**
         * Randomly splits the available examples into two distinct sets and updates a given `BiPartition` accordingly.
         *
         * @param partition A reference to an object of type `BiPartition` to be updated
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         */
        void sampleBiPartition(BiPartition& partition, RNG& rng) const;
};
