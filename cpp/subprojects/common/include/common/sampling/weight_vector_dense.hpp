/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dense.hpp"
#include "common/sampling/weight_vector.hpp"


/**
 * An one-dimensional vector that provides random access to a fixed number of weights stored in a C-contiguous array.
 *
 * @tparam T The type of the weights
 */
template<typename T>
class DenseWeightVector final : public IWeightVector {

    private:

        DenseVector<T> vector_;

        uint32 numNonZeroWeights_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseWeightVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseWeightVector(uint32 numElements, bool init);

        /**
         * An iterator that provides access to the weights in the vector and allows to modify them.
         */
        typedef typename DenseVector<T>::iterator iterator;

        /**
         * An iterator that provides read-only access to the weights in the vector.
         */
        typedef typename DenseVector<T>::const_iterator const_iterator;

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

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        uint32 getNumElements() const;

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
