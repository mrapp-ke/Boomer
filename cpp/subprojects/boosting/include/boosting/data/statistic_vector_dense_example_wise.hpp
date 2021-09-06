/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "boosting/iterator/diagonal_iterator.hpp"
#include <memory>


namespace boosting {

    // Forward declarations
    class IExampleWiseRuleEvaluationFactory;
    template<typename StatisticVector>
    class IRuleEvaluation;

    /**
     * An one-dimensional vector that stores gradients and Hessians that have been calculated using a non-decomposable
     * loss function in C-contiguous arrays. For each element in the vector a single gradient, but multiple Hessians are
     * stored. In a vector that stores `n` gradients `(n * (n + 1)) / 2` Hessians are stored. The Hessians can be viewed
     * as a symmetric Hessian matrix with `n` rows and columns.
     */
    class DenseExampleWiseStatisticVector final {

        private:

            uint32 numGradients_;

            uint32 numHessians_;

            float64* gradients_;

            float64* hessians_;

        public:

            /**
             * @param numGradients The number of gradients in the vector
             */
            DenseExampleWiseStatisticVector(uint32 numGradients);

            /**
             * @param numGradients The number of gradients in the vector
             * @param init         True, if all gradients and Hessians in the vector should be initialized with zero,
             *                     false otherwise
             */
            DenseExampleWiseStatisticVector(uint32 numGradients, bool init);

            /**
             * @param vector A reference to an object of type `DenseExampleWiseStatisticVector` to be copied
             */
            DenseExampleWiseStatisticVector(const DenseExampleWiseStatisticVector& vector);

            ~DenseExampleWiseStatisticVector();

            /**
             * An iterator that provides access to the gradients in the vector and allows to modify them.
             */
            typedef float64* gradient_iterator;

            /**
             * An iterator that provides read-only access to the gradients in the vector.
             */
            typedef const float64* gradient_const_iterator;

            /**
             * An iterator that provides access to the Hessians in the vector and allows to modify them.
             */
            typedef float64* hessian_iterator;

            /**
             * An iterator that provides read-only access to the Hessians in the vector.
             */
            typedef const float64* hessian_const_iterator;

            /**
             * An iterator that provides read-only access to the Hessians that correspond to the diagonal of the Hessian
             * matrix.
             */
            typedef DiagonalConstIterator<float64> hessian_diagonal_const_iterator;

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_iterator` to the beginning
             */
            gradient_iterator gradients_begin();

            /**
             * Returns a `gradient_iterator` to the end of the gradients.
             *
             * @return A `gradient_iterator` to the end
             */
            gradient_iterator gradients_end();

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_const_iterator` to the beginning
             */
            gradient_const_iterator gradients_cbegin() const;

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients.
             *
             * @return A `gradient_const_iterator` to the end
             */
            gradient_const_iterator gradients_cend() const;

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_iterator` to the beginning
             */
            hessian_iterator hessians_begin();

            /**
             * Returns a `hessian_iterator` to the end of the Hessians.
             *
             * @return A `hessian_iterator` to the end
             */
            hessian_iterator hessians_end();

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_const_iterator` to the beginning
             */
            hessian_const_iterator hessians_cbegin() const;

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians.
             *
             * @return A `hessian_const_iterator` to the end
             */
            hessian_const_iterator hessians_cend() const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the beginning of the Hessians that correspond to the
             * diagonal of the Hessian matrix.
             *
             * @return A `hessian_diagonal_const_iterator` to the beginning
             */
            hessian_diagonal_const_iterator hessians_diagonal_cbegin() const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the end of the Hessians that correspond to the diagonal of
             * the Hessian matrix.
             *
             * @return A `hessian_diagonal_const_iterator` to the end
             */
            hessian_diagonal_const_iterator hessians_diagonal_cend() const;

            /**
             * Returns the number of gradients in the vector.
             *
             * @return The number of gradients
             */
            uint32 getNumElements() const;

            /**
             * Sets all gradients and Hessians in the vector to zero.
             */
            void clear();

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             */
            void add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                     hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd);

            /**
             * Adds all gradients and Hessians in another vector to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                     hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd, float64 weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a
             * specific weight.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             * @param indices           A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                             hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a specific
             * weight.
             *
             * @param gradientsBegin    A `gradient_const_iterator` to the beginning of the gradients
             * @param gradientsEnd      A `gradient_const_iterator` to the end of the gradients
             * @param hessiansBegin     A `hessian_const_iterator` to the beginning of the Hessians
             * @param hessiansEnd       A `hessian_const_iterator` to the end of the Hessians
             * @param indices           A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight            The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                             hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param firstGradientsBegin   A `gradient_const_iterator` to the beginning of the first gradients
             * @param firstGradientsEnd     A `gradient_const_iterator` to the end of the first gradients
             * @param firstHessiansBegin    A `hessian_const_iterator` to the beginning of the first Hessians
             * @param firstHessiansEnd      A `hessian_const_iterator` to the end of the first Hessians
             * @param firstIndices          A reference to an object of type `CompleteIndexVector` that provides access
             *                              to the indices
             * @param secondGradientsBegin  A `gradient_const_iterator` to the beginning of the second gradients
             * @param secondGradientsEnd    A `gradient_const_iterator` to the end of the second gradients
             * @param secondHessiansBegin   A `hessian_const_iterator` to the beginning of the second Hessians
             * @param secondHessiansEnd     A `hessian_const_iterator` to the end of the second Hessians
             */
            void difference(gradient_const_iterator firstGradientsBegin, gradient_const_iterator firstGradientsEnd,
                            hessian_const_iterator firstHessiansBegin, hessian_const_iterator firstHessiansEnd,
                            const CompleteIndexVector& firstIndices, gradient_const_iterator secondGradientsBegin,
                            gradient_const_iterator secondGradientsEnd, hessian_const_iterator secondHessiansBegin,
                            hessian_const_iterator secondHessiansEnd);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param firstGradientsBegin   A `gradient_const_iterator` to the beginning of the first gradients
             * @param firstGradientsEnd     A `gradient_const_iterator` to the end of the first gradients
             * @param firstHessiansBegin    A `hessian_const_iterator` to the beginning of the first Hessians
             * @param firstHessiansEnd      A `hessian_const_iterator` to the end of the first Hessians
             * @param firstIndices          A reference to an object of type `PartialIndexVector` that provides access
             *                              to the indices
             * @param secondGradientsBegin  A `gradient_const_iterator` to the beginning of the second gradients
             * @param secondGradientsEnd    A `gradient_const_iterator` to the end of the second gradients
             * @param secondHessiansBegin   A `hessian_const_iterator` to the beginning of the second Hessians
             * @param secondHessiansEnd     A `hessian_const_iterator` to the end of the second Hessians
             */
            void difference(gradient_const_iterator firstGradientsBegin, gradient_const_iterator firstGradientsEnd,
                            hessian_const_iterator firstHessiansBegin, hessian_const_iterator firstHessiansEnd,
                            const PartialIndexVector& firstIndices, gradient_const_iterator secondGradientsBegin,
                            gradient_const_iterator secondGradientsEnd, hessian_const_iterator secondHessiansBegin,
                            hessian_const_iterator secondHessiansEnd);

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules, based on the gradients and Hessians that are stored in a `DenseExampleWiseStatisticVector`.
             *
             * @param factory       A reference to an object of type `IExampleWiseRuleEvaluationFactory` that should be
             *                      used to create the object
             * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> createRuleEvaluation(
                const IExampleWiseRuleEvaluationFactory& factory, const CompleteIndexVector& labelIndices) const;

            /**
             * Creates and returns a new object of type `IRuleEvaluation` that allows to calculate the predictions of
             * rules, based on the gradients and Hessians that are stored in a `DenseExampleWiseStatisticVector`.
             *
             * @param factory       A reference to an object of type `IExampleWiseRuleEvaluationFactory` that should be
             *                      used to create the object
             * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `IRuleEvaluation` that has been created
             */
            std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> createRuleEvaluation(
                const IExampleWiseRuleEvaluationFactory& factory, const PartialIndexVector& labelIndices) const;

    };

}
