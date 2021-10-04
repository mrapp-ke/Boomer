/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics.hpp"
#include <functional>


namespace seco {

    // Forward declarations
    class DenseWeightMatrix;

    /**
     * Defines an interface for all classes that allow to store the elements of confusion matrices that have been
     * computed based on a weight matrix and the ground truth labels of the training examples.
     */
    class ICoverageStatistics : public IStatistics {

        public:

            virtual ~ICoverageStatistics() { };

            /**
             * A visitor function for handling objects of the type `DenseWeightMatrix`.
             */
            typedef std::function<void(std::unique_ptr<DenseWeightMatrix>&)> DenseWeightMatrixVisitor;

            /**
             * Invokes one of the given visitor functions, depending on which one is able to handle the particular type
             * of matrix that is used to store the weights of individual examples and labels.
             *
             * @param denseWeightMatrixVisitor The visitor function for handling objects of the type `DenseWeightMatrix`
             */
            virtual void visitWeightMatrix(DenseWeightMatrixVisitor denseWeightMatrixVisitor) = 0;

            /**
             * Returns the sum of the weights of all labels that remain to be covered.
             *
             * @return The sum of the weights
             */
            virtual float64 getSumOfUncoveredWeights() const = 0;

    };

}
