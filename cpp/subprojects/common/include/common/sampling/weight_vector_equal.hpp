/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/weight_vector.hpp"


/**
 * An one-dimensional vector that provides random access to a fixed number of equal weights.
 */
class EqualWeightVector final : public IWeightVector {

    private:

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        EqualWeightVector(uint32 numElements);

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        uint32 getNumElements() const;

        uint32 getNumNonZeroWeights() const override;

        bool hasZeroWeights() const override;

        float64 getWeight(uint32 pos) const override;

};
