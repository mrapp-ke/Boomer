/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dense.hpp"
#include "common/sampling/weight_vector.hpp"


/**
 * An one-dimensional vector that provides random access to a fixed number of weights stored in a C-contiguous array.
 */
class DenseWeightVector final : public IWeightVector {

    private:

        DenseVector<uint32> vector_;

        uint32 sumOfWeights_;

    public:

        /**
         * @param numElements   The number of elements in the vector. Must be at least 1
         * @param sumOfWeights  The sum of the weights in the vector
         */
        DenseWeightVector(uint32 numElements, uint32 sumOfWeights);

        /**
         * An iterator that provides access to the weights in the vector and allows to modify them.
         */
        typedef DenseVector<uint32>::iterator iterator;

        /**
         * An iterator that provides read-only access to the weights in the vector.
         */
        typedef DenseVector<uint32>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        bool hasZeroWeights() const override;

        uint32 getWeight(uint32 pos) const override;

        uint32 getSumOfWeights() const override;

};
