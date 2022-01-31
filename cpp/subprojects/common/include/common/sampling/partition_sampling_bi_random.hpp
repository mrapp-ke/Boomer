/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to configure a method for partitioning the available training
 * examples into a training set and a holdout set that randomly splits the training examples into two mutually exclusive
 * sets.
 */
class MLRLCOMMON_API IRandomBiPartitionSamplingConfig {

    public:

        virtual ~IRandomBiPartitionSamplingConfig() { };

        /**
         * Returns the fraction of examples that are included in the holdout set.
         *
         * @return The fraction of examples that are included in the holdout set
         */
        virtual float32 getHoldoutSetSize() const = 0;

        /**
         * Sets the fraction of examples that should be included in the holdout set.
         *
         * @param holdoutSetSize    The fraction of examples that should be included in the holdout set, e.g. a value of
         *                          0.6 corresponds to 60 % of the available examples. Must be in (0, 1)
         * @return                  A reference to an object of type `IRandomBiPartitionSamplingConfig` that allows
         *                          further configuration of the method for partitioning the available training examples
         *                          into a training set and a holdout set
         */
        virtual IRandomBiPartitionSamplingConfig& setHoldoutSetSize(float32 holdoutSetSize) = 0;

};

/**
 * Allows to configure a method for partitioning the available training examples into a training set and a holdout set
 * that randomly splits the training examples into two mutually exclusive sets.
 */
class RandomBiPartitionSamplingConfig final : public IPartitionSamplingConfig, public IRandomBiPartitionSamplingConfig {

    private:

        float32 holdoutSetSize_;

    public:

        RandomBiPartitionSamplingConfig();

        float32 getHoldoutSetSize() const override;

        IRandomBiPartitionSamplingConfig& setHoldoutSetSize(float32 holdoutSetSize) override;

        std::unique_ptr<IPartitionSamplingFactory> createPartitionSamplingFactory() const override;

};
