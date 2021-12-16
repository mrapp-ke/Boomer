/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/weight_vector.hpp"
#include "common/sampling/random.hpp"
#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "common/statistics/statistics.hpp"
#include <memory>

// Forward declarations
class BiPartition;
class SinglePartition;


/**
 * Defines an interface for all classes that implement a strategy for sampling training examples.
 */
class IInstanceSampling {

    public:

        virtual ~IInstanceSampling() { };

        /**
         * Creates and returns a sample of the available training examples.
         *
         * @param rng       A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return          A reference to an object type `WeightVector` that provides access to the weights of the
         *                  individual training examples
         */
        virtual const IWeightVector& sample(RNG& rng) = 0;

};


/**
 * Defines an interface for all factories that allow to create instances of the type `IInstanceSampling`.
 */
class IInstanceSamplingFactory {

    public:

        virtual ~IInstanceSamplingFactory() { };

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides access to the
         *                      labels of the training examples
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics +
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CContiguousLabelMatrix& labelMatrix,
                                                          const SinglePartition& partition,
                                                          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides access to the
         *                      labels of the training examples
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics +
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CContiguousLabelMatrix& labelMatrix,
                                                          BiPartition& partition, IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides access to the labels of
         *                      the training examples
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics +
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CsrLabelMatrix& labelMatrix,
                                                          const SinglePartition& partition,
                                                          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new object of type `IInstanceSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides access to the labels of
         *                      the training examples
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics +
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> create(const CsrLabelMatrix& labelMatrix,
                                                          BiPartition& partition, IStatistics& statistics) const = 0;

};
