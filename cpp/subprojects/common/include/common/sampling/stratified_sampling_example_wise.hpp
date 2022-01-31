/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/partition_bi.hpp"
#include <unordered_map>
#include <vector>
#include <functional>


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
        ExampleWiseStratification(const LabelMatrix& labelMatrix, IndexIterator indicesBegin, IndexIterator indicesEnd);

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
