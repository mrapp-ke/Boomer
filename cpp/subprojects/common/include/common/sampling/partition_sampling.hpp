/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/partition.hpp"
#include "common/sampling/random.hpp"
#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include <memory>


/**
 * Defines an interface for all classes that implement a strategy for partitioning the available training examples into
 * a training set and a holdout set.
 */
class IPartitionSampling {

    public:

        virtual ~IPartitionSampling() { };

        /**
         * Creates and returns a partition of the available training examples.
         *
         * @param rng   A reference to an object of type `RNG`, implementing the random number generator to be used
         * @return      A reference to an object of type `IPartition` that provides access to the indices of the
         *              training examples that belong to the training set and holdout set, respectively
         */
        virtual IPartition& partition(RNG& rng) = 0;

};

/**
 * Defines an interface for all factories that allow to create objects of type `IPartitionSampling`.
 */
class IPartitionSamplingFactory {

    public:

        virtual ~IPartitionSamplingFactory() { };

        /**
         * Creates and returns a new object of type `IPartitionSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides random access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> create(const CContiguousLabelMatrix& labelMatrix) const = 0;

        /**
         * Creates and returns a new object of type `IPartitionSampling`.
         *
         * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides row-wise access to the
         *                      labels of the training examples
         * @return              An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> create(const CsrLabelMatrix& labelMatrix) const = 0;

};
