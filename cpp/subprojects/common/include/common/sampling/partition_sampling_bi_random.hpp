/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition_sampling.hpp"


/**
 * Allows to create objects of the type `IPartitionSampling` that randomly split the training examples into two mutually
 * exclusive sets that may be used as a training set and a holdout set.
 */
class RandomBiPartitionSamplingFactory final : public IPartitionSamplingFactory {

    private:

        float32 holdoutSetSize_;

    public:

        /**
         * @param holdoutSetSize The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                       corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        RandomBiPartitionSamplingFactory(float32 holdoutSetSize);

        std::unique_ptr<IPartitionSampling> create(const CContiguousLabelMatrix& labelMatrix) const override;

        std::unique_ptr<IPartitionSampling> create(const CsrLabelMatrix& labelMatrix) const override;

};
