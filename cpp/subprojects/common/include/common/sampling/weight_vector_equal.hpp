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
         * @param numElements The number of elements in the vector. Must be at least 1
         */
        EqualWeightVector(uint32 numElements);

        bool hasZeroWeights() const override;

        uint32 getWeight(uint32 pos) const override;

        uint32 getSumOfWeights() const override;

};
