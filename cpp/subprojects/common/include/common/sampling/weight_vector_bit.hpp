/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_bit.hpp"
#include "common/sampling/weight_vector.hpp"


/**
 * An one-dimensional vector that provides random access to a fixed number of binary weights stored in a `BitVector`.
 */
class BitWeightVector final : public IWeightVector {

    private:

        BitVector vector_;

        uint32 numNonZeroWeights_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        BitWeightVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        BitWeightVector(uint32 numElements, bool init);

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        uint32 getNumElements() const;

        /**
         * Sets the weight at a specific position.
         *
         * @param pos       The position
         * @param weight    The weight to be set
         */
        void set(uint32 pos, bool weight);

        /**
         * Sets all weights to zero.
         */
        void clear();

        /**
         * Sets the number of non-zero weights.
         *
         * @param numNonZeroWeights The number of non-zero weights to be set
         */
        void setNumNonZeroWeights(uint32 numNonZeroWeights);

        uint32 getNumNonZeroWeights() const override;

        bool hasZeroWeights() const override;

        float64 getWeight(uint32 pos) const override;

};
