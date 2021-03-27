/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dok_binary.hpp"
#include <memory>


/**
 * An one-dimensional sparse vector that stores the indices of training examples with missing feature values using the
 * dictionary of keys (DOK) format.
 */
class MissingFeatureVector {

    private:

        std::unique_ptr<BinaryDokVector> missingIndicesPtr_;

    public:

        MissingFeatureVector();

        /**
         * @param missingFeatureVector A reference to an object of type `MissingFeatureVector`, the missing indices
         *                             should be taken from
         */
        MissingFeatureVector(MissingFeatureVector& missingFeatureVector);

        /**
         * An iterator that provides read-only access to the missing indices.
         */
        typedef BinaryDokVector::index_const_iterator missing_index_const_iterator;

        /**
         * Returns a `missing_index_const_iterator` to the beginning of the missing indices.
         *
         * @return A `missing_index_const_iterator` to the beginning
         */
        missing_index_const_iterator missing_indices_cbegin() const;

        /**
         * Returns a `missing_index_const_iterator` to the end of the missing indices.
         *
         * @return A `missing_index_const_iterator` to the end
         */
        missing_index_const_iterator missing_indices_cend() const;

        /**
         * Adds the index of an example with missing feature value.
         *
         * @param index The index to be added
         */
        void addMissingIndex(uint32 index);

        /**
         * Returns whether the example at a specific index has a missing feature value.
         *
         * @param index The index of the example to be checked
         * @return      True, if the example at the given index has a missing feature value, false otherwise
         */
        bool isMissing(uint32 index) const;

        /**
         * Removes all indices of examples with missing feature values.
         */
        void clearMissingIndices();

};
