/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/tuple.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include <memory>



namespace boosting {

    // Forward declarations
    class ILabelWiseRuleEvaluationFactory;
    template<typename StatisticVector>
    class IRuleEvaluation;

    /**
     * An one-dimensional vector that stores gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function in C-contiguous arrays. For each element in the vector a single gradient and Hessian
     * is stored.
     */
    class DenseLabelWiseStatisticVector final {

        private:

            uint32 numElements_;

            Tuple<float64>* statistics_;

        public:

            /**
             * @param numElements The number of gradients and Hessians in the vector
             */
            DenseLabelWiseStatisticVector(uint32 numElements);

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             * @param init          True, if all gradients and Hessians in the vector should be initialized with zero,
             *                      false otherwise
             */
            DenseLabelWiseStatisticVector(uint32 numElements, bool init);

            /**
             * @param vector A reference to an object of type `DenseLabelWiseStatisticVector` to be copied
             */
            DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& vector);

            ~DenseLabelWiseStatisticVector();

            /**
             * An iterator that provides access to the elements in the vector and allows to modify them.
             */
            typedef Tuple<float64>* iterator;

            /**
             * An iterator that provides read-only access to the elements in the vector.
             */
            typedef const Tuple<float64>* const_iterator;

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
             * Returns the number of gradients and Hessians in the vector.
             *
             * @return The number of gradients and Hessians
             */
            uint32 getNumElements() const;

            /**
             * Sets all gradients and Hessians in the vector to zero.
             */
            void clear();

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param begin A `const_iterator` to the beginning of the vector
             * @param end   A `const_iterator` to the end of the vector
             */
            void add(const_iterator begin, const_iterator end);

            /**
             * Adds all gradients and Hessians in another vector to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param begin     A `const_iterator` to the beginning of the vector
             * @param end       A `const_iterator` to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const_iterator begin, const_iterator end, float64 weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a
             * specific weight.
             *
             * @param begin     A `const_iterator` to the beginning of the vector
             * @param end       A `const_iterator` to the end of the vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const_iterator begin, const_iterator end, const CompleteIndexVector& indices,
                             float64 weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a specific
             * weight.
             *
             * @param begin     A `const_iterator` to the beginning of the vector
             * @param end       A `const_iterator` to the end of the vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const_iterator begin, const_iterator end, const PartialIndexVector& indices,
                             float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param firstBegin    A `const_iterator` to the beginning of the first vector
             * @param firstEnd      A `const_iterator` to the end of the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param secondBegin  A `const_iterator` to the beginning of the second vector
             * @param secondEnd    A `const_iterator` to the end of the second vector
             */
            void difference(const_iterator firstBegin, const_iterator firstEnd, const CompleteIndexVector& firstIndices,
                            const_iterator secondBegin, const_iterator secondEnd);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param firstBegin    A `const_iterator` to the beginning of the first vector
             * @param firstEnd      A `const_iterator` to the end of the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param secondBegin   A `const_iterator` to the beginning of the second vector
             * @param secondEnd     A `const_iterator` to the end of the second vector
             */
            void difference(const_iterator firstBegin, const_iterator firstEnd, const PartialIndexVector& firstIndices,
                            const_iterator secondBegin, const_iterator secondEnd);

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules, based on the gradients and Hessians that are stored in a `DenseLabelWiseStatisticVector`.
             *
             * @param factory       A reference to an object of type `ILabelWiseRuleEvaluationFactory` that should be
             *                      used to create the object
             * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> createRuleEvaluation(
                const ILabelWiseRuleEvaluationFactory& factory, const CompleteIndexVector& labelIndices) const;

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules, based on the gradients and Hessians that are stored in a `DenseLabelWiseStatisticVector`.
             *
             * @param factory       A reference to an object of type `ILabelWiseRuleEvaluationFactory` that should be
             *                      used to create the object
             * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> createRuleEvaluation(
                const ILabelWiseRuleEvaluationFactory& factory, const PartialIndexVector& labelIndices) const;

    };

}
