/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/weight_vector.hpp"

/**
 * An one-dimensional vector that provides random access to a fixed number of equal weights.
 */
class EqualWeightVector final : public IWeightVector {
    private:

        const uint32 numElements_;

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

        /**
         * Returns the number of non-zero weights.
         *
         * @return The number of non-zero weights
         */
        uint32 getNumNonZeroWeights() const;

        /**
         * Returns the weight at a specific position.
         *
         * @param pos   The position
         * @return      The weight at the specified position
         */
        uint32 operator[](uint32 pos) const;

        bool hasZeroWeights() const override;

        std::unique_ptr<IThresholdsSubset> createThresholdsSubset(IThresholds& thresholds) const override;
};
